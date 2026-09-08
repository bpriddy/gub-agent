/**
 * batch/auth.ts — where the GUB JWT a batch runs under comes from.
 *
 * Two sources, chosen per run (`run_as`):
 *
 *   `me`      — the signed-in user's token, pushed by the browser: the page
 *               polls progress with a fresh `Authorization` header, and the
 *               run picks up the newest one. If the tab closes the run lives
 *               until that token expires (15 min), then stops on auth.
 *   `subject` — the DEDICATED SANDBOX SUBJECT (epic invariant 10): the server
 *               impersonates `SANDBOX_JWT_SUBJECT_SA` with its own ADC
 *               (roles/iam.serviceAccountTokenCreator on that SA), exchanges the
 *               impersonated Google access token at GUB's
 *               /auth/google/access-token-exchange, rotates the 15-minute GUB
 *               token via /auth/refresh and revokes the session at the end.
 *               A distinct GUB user → its own per-subject rate bucket, and
 *               batch traffic is filterable in the backend logs by that email.
 *               The SA needs NO GCP role: Vertex is still called with the
 *               operator's ADC.
 *
 * No token is ever written to disk.
 */
import { GoogleAuth, Impersonated } from 'google-auth-library';

const JWT_REFRESH_MARGIN_S = 120;
const USERINFO_SCOPE = 'https://www.googleapis.com/auth/userinfo.email';

export function gubBaseUrl(): string {
  const url = process.env.GUB_BACKEND_URL?.trim();
  if (!url) throw new Error('GUB_BACKEND_URL is not set in .env.local');
  return url.replace(/\/$/, '');
}

export function subjectServiceAccount(): string | null {
  return process.env.SANDBOX_JWT_SUBJECT_SA?.trim() || null;
}

export function decodeJwtPayload(jwt: string): Record<string, unknown> | null {
  try {
    const payload = jwt.split('.')[1];
    if (!payload) return null;
    return JSON.parse(Buffer.from(payload, 'base64url').toString('utf8')) as Record<string, unknown>;
  } catch {
    return null;
  }
}

export class JwtExpiredError extends Error {
  code = 'jwt-expired' as const;
}

export interface TokenSource {
  readonly kind: 'me' | 'subject';
  /** Email or sub of the identity, for run.json and the UI. */
  subject(): string | null;
  secondsLeft(): number | null;
  canRenew(): boolean;
  /** A usable token, renewing first when about to expire. Throws JwtExpiredError when it cannot. */
  token(): Promise<string>;
  /** Force a renewal (after an auth failure). Returns false when this source cannot renew. */
  renew(): Promise<boolean>;
  /** Called once when the run ends. */
  close(): Promise<void>;
}

function secondsLeftOf(jwt: string | null): number | null {
  const exp = jwt ? decodeJwtPayload(jwt)?.exp : null;
  return typeof exp === 'number' ? exp - Math.floor(Date.now() / 1000) : null;
}

function subjectOf(jwt: string | null): string | null {
  const p = jwt ? decodeJwtPayload(jwt) : null;
  const email = p?.email;
  const sub = p?.sub;
  return typeof email === 'string' ? email : typeof sub === 'string' ? sub : null;
}

/** The browser's token. `update()` is called on every progress poll. */
export class MeTokenSource implements TokenSource {
  readonly kind = 'me' as const;
  private jwt: string;
  constructor(jwt: string) { this.jwt = jwt; }
  update(jwt: string): void {
    // Only ever move forward: a stale poll must not roll the token back.
    const now = secondsLeftOf(this.jwt) ?? -1;
    const next = secondsLeftOf(jwt) ?? -1;
    if (next >= now) this.jwt = jwt;
  }
  subject() { return subjectOf(this.jwt); }
  secondsLeft() { return secondsLeftOf(this.jwt); }
  canRenew() { return false; }
  async token() {
    const left = this.secondsLeft();
    if (left !== null && left <= 0) throw new JwtExpiredError(`the signed-in user's GUB token expired at ${new Date((decodeJwtPayload(this.jwt)!.exp as number) * 1000).toISOString()} and the page stopped pushing fresh ones`);
    return this.jwt;
  }
  async renew() { return false; }
  async close() { /* the browser owns this session */ }
}

interface ExchangeResponse { accessToken?: string; refreshToken?: string; expiresIn?: number }

/** The dedicated sandbox subject: impersonate → exchange → refresh → logout. */
export class SubjectTokenSource implements TokenSource {
  readonly kind = 'subject' as const;
  private jwt: string | null = null;
  private refreshToken: string | null = null;
  private renewing: Promise<void> | null = null;
  readonly log: string[] = [];
  constructor(private readonly serviceAccount: string, private readonly baseUrl: string) {}

  static fromEnv(): SubjectTokenSource {
    const sa = subjectServiceAccount();
    if (!sa) throw new Error('SANDBOX_JWT_SUBJECT_SA is not set in .env.local — no dedicated sandbox subject is configured');
    return new SubjectTokenSource(sa, gubBaseUrl());
  }

  subject() { return subjectOf(this.jwt) ?? this.serviceAccount; }
  secondsLeft() { return secondsLeftOf(this.jwt); }
  canRenew() { return true; }

  async token(): Promise<string> {
    const left = this.secondsLeft();
    if (this.jwt === null || (left !== null && left < JWT_REFRESH_MARGIN_S)) await this.renewOnce();
    return this.jwt!;
  }

  async renew(): Promise<boolean> {
    await this.renewOnce();
    return true;
  }

  private renewOnce(): Promise<void> {
    if (!this.renewing) {
      this.renewing = (async () => {
        try {
          if (this.refreshToken) {
            try { await this.refresh(); return; } catch (e) { this.log.push(`refresh failed (${(e as Error).message}) — re-exchanging`); }
          }
          await this.exchange();
        } finally {
          this.renewing = null;
        }
      })();
    }
    return this.renewing;
  }

  /** Impersonated Google access token carrying the SA's email (what GUB's userinfo lookup needs). */
  static async impersonatedAccessToken(serviceAccount: string): Promise<string> {
    const auth = new GoogleAuth({ scopes: ['https://www.googleapis.com/auth/cloud-platform'] });
    const source = await auth.getClient();
    const client = new Impersonated({
      sourceClient: source,
      targetPrincipal: serviceAccount,
      targetScopes: [USERINFO_SCOPE],
      lifetime: 3600,
    });
    const { token } = await client.getAccessToken();
    if (!token) throw new Error(`could not mint an impersonated token for ${serviceAccount} — does your ADC identity hold roles/iam.serviceAccountTokenCreator on it?`);
    return token;
  }

  private async exchange(): Promise<void> {
    const googleToken = await SubjectTokenSource.impersonatedAccessToken(this.serviceAccount);
    const res = await fetch(`${this.baseUrl}/auth/google/access-token-exchange`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ accessToken: googleToken }), signal: AbortSignal.timeout(20_000),
    });
    if (!res.ok) throw new JwtExpiredError(`GUB token exchange for ${this.serviceAccount} failed: ${res.status} ${(await res.text()).slice(0, 300)}`);
    const data = (await res.json()) as ExchangeResponse;
    if (!data.accessToken) throw new JwtExpiredError('GUB token exchange returned no accessToken');
    this.jwt = data.accessToken;
    this.refreshToken = data.refreshToken ?? null;
    this.log.push(`minted GUB JWT for ${this.subject()} (valid ${this.secondsLeft()}s)`);
  }

  private async refresh(): Promise<void> {
    const res = await fetch(`${this.baseUrl}/auth/refresh`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refreshToken: this.refreshToken }), signal: AbortSignal.timeout(20_000),
    });
    if (!res.ok) throw new Error(`POST /auth/refresh → ${res.status}`);
    const data = (await res.json()) as ExchangeResponse;
    if (!data.accessToken) throw new Error('POST /auth/refresh: no accessToken');
    this.jwt = data.accessToken;
    if (data.refreshToken) this.refreshToken = data.refreshToken;
    this.log.push(`GUB JWT rotated via /auth/refresh (valid ${this.secondsLeft()}s)`);
  }

  async close(): Promise<void> {
    if (!this.refreshToken) return;
    try {
      const res = await fetch(`${this.baseUrl}/auth/logout`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refreshToken: this.refreshToken }), signal: AbortSignal.timeout(10_000),
      });
      this.log.push(`sandbox-subject GUB session revoked (${res.status})`);
    } catch (e) {
      this.log.push(`could not revoke the sandbox-subject session: ${(e as Error).message}`);
    }
    this.refreshToken = null;
  }
}

/** GET /api/batch/subject — is a dedicated subject configured, and can this server mint for it? */
export async function probeSubject(): Promise<{ configured: boolean; serviceAccount: string | null; ok: boolean; email: string | null; error: string | null }> {
  const sa = subjectServiceAccount();
  if (!sa) return { configured: false, serviceAccount: null, ok: false, email: null, error: null };
  try {
    const token = await SubjectTokenSource.impersonatedAccessToken(sa);
    const res = await fetch(`https://oauth2.googleapis.com/tokeninfo?access_token=${encodeURIComponent(token)}`, { signal: AbortSignal.timeout(10_000) });
    const info = (await res.json()) as { email?: string; error_description?: string };
    if (!res.ok) return { configured: true, serviceAccount: sa, ok: false, email: null, error: info.error_description ?? `tokeninfo ${res.status}` };
    return { configured: true, serviceAccount: sa, ok: true, email: info.email ?? null, error: null };
  } catch (e) {
    return { configured: true, serviceAccount: sa, ok: false, email: null, error: (e as Error).message };
  }
}
