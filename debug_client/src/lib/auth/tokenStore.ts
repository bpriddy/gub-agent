/**
 * tokenStore.ts — access token in memory, refresh token in localStorage.
 * Ported from the GUB frontend. Browser-only (guards window access).
 */
import { refreshTokens } from './authClient';
import type { AuthResponse } from './authClient';

const REFRESH_TOKEN_KEY = 'gub_debug_refresh_token';
const PROACTIVE_REFRESH_BUFFER_MS = 60_000;

interface AccessTokenState {
  token: string;
  expiresAt: number;
  user: AuthResponse['user'];
}

let accessTokenState: AccessTokenState | null = null;

export function setSession(response: AuthResponse): void {
  accessTokenState = {
    token: response.accessToken,
    expiresAt: Date.now() + response.expiresIn * 1000,
    user: response.user,
  };
  if (typeof window !== 'undefined') {
    localStorage.setItem(REFRESH_TOKEN_KEY, response.refreshToken);
  }
}

export function clearSession(): void {
  accessTokenState = null;
  if (typeof window !== 'undefined') {
    localStorage.removeItem(REFRESH_TOKEN_KEY);
  }
}

export function getRefreshToken(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem(REFRESH_TOKEN_KEY);
}

export function getUser(): AuthResponse['user'] | null {
  return accessTokenState?.user ?? null;
}

export function isAccessTokenValid(): boolean {
  if (!accessTokenState) return false;
  return Date.now() < accessTokenState.expiresAt - PROACTIVE_REFRESH_BUFFER_MS;
}

export function isLoggedIn(): boolean {
  return getRefreshToken() !== null;
}

let refreshPromise: Promise<string> | null = null;

export async function getValidAccessToken(): Promise<string> {
  if (isAccessTokenValid() && accessTokenState) {
    return accessTokenState.token;
  }
  if (refreshPromise) return refreshPromise;

  refreshPromise = (async () => {
    const refreshToken = getRefreshToken();
    if (!refreshToken) throw new Error('No refresh token available — user must log in');
    const response = await refreshTokens(refreshToken);
    setSession(response);
    return response.accessToken;
  })().finally(() => {
    refreshPromise = null;
  });

  return refreshPromise;
}
