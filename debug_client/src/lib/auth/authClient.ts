/**
 * authClient.ts — talk to GUB's auth endpoints. Ported from the GUB
 * frontend; BASE_URL is `/gub` (Next rewrites it to the GUB backend so the
 * calls are same-origin and CORS-free).
 */

const BASE_URL = '/gub';

export interface AuthResponse {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
  tokenType: 'Bearer';
  user: {
    id: string;
    email: string;
    displayName: string | null;
    avatarUrl: string | null;
  };
}

export interface ApiError {
  code: string;
  message: string;
  details?: unknown;
}

export class AuthApiError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = 'AuthApiError';
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (res.ok) {
    if (res.status === 204) return undefined as T;
    return res.json() as Promise<T>;
  }
  let body: ApiError = { code: 'UNKNOWN_ERROR', message: 'An unknown error occurred' };
  try {
    body = (await res.json()) as ApiError;
  } catch {
    /* ignore parse failure */
  }
  throw new AuthApiError(body.code, body.message, res.status);
}

export async function loginWithGoogle(idToken: string): Promise<AuthResponse> {
  const res = await fetch(`${BASE_URL}/auth/google/exchange`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ idToken }),
  });
  return handleResponse<AuthResponse>(res);
}

export async function refreshTokens(refreshToken: string): Promise<AuthResponse> {
  const res = await fetch(`${BASE_URL}/auth/refresh`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refreshToken }),
  });
  return handleResponse<AuthResponse>(res);
}

export async function logout(refreshToken: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/auth/logout`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refreshToken }),
  });
  return handleResponse<void>(res);
}
