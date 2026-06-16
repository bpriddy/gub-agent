/**
 * useAuth.ts — Google Identity Services → GUB exchange → session state.
 * Ported from the GUB frontend; client_id from NEXT_PUBLIC_GOOGLE_CLIENT_ID.
 */
'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { loginWithGoogle, logout as apiLogout, AuthApiError } from './authClient';
import {
  setSession,
  clearSession,
  getRefreshToken,
  getValidAccessToken,
  getUser,
  isLoggedIn,
} from './tokenStore';
import type { AuthResponse } from './authClient';

export type AuthStatus = 'loading' | 'authenticated' | 'unauthenticated' | 'error';

export interface AuthState {
  status: AuthStatus;
  user: AuthResponse['user'] | null;
  error: string | null;
}

export interface UseAuthReturn extends AuthState {
  renderGoogleButton: (container: HTMLElement) => void;
  logout: () => Promise<void>;
}

const GOOGLE_CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID as string;

export function useAuth(): UseAuthReturn {
  const [state, setState] = useState<AuthState>({ status: 'loading', user: null, error: null });
  const restoredRef = useRef(false);

  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;

    if (!isLoggedIn()) {
      setState({ status: 'unauthenticated', user: null, error: null });
      return;
    }
    getValidAccessToken()
      .then(() => setState({ status: 'authenticated', user: getUser(), error: null }))
      .catch(() => {
        clearSession();
        setState({ status: 'unauthenticated', user: null, error: null });
      });
  }, []);

  const handleGoogleCredential = useCallback(
    async (credentialResponse: google.accounts.id.CredentialResponse) => {
      setState((s) => ({ ...s, status: 'loading', error: null }));
      try {
        const response = await loginWithGoogle(credentialResponse.credential);
        setSession(response);
        setState({ status: 'authenticated', user: response.user, error: null });
      } catch (err) {
        const message = err instanceof AuthApiError ? err.message : 'Login failed — please try again';
        clearSession();
        setState({ status: 'error', user: null, error: message });
      }
    },
    [],
  );

  const renderGoogleButton = useCallback(
    (container: HTMLElement) => {
      if (!window.google?.accounts?.id) {
        console.warn('Google Identity Services not yet loaded');
        return;
      }
      window.google.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID,
        callback: (credentialResponse) => void handleGoogleCredential(credentialResponse),
        cancel_on_tap_outside: false,
      });
      window.google.accounts.id.renderButton(container, {
        type: 'standard',
        theme: 'filled_black',
        size: 'large',
        text: 'signin_with',
        shape: 'rectangular',
        width: 280,
      });
    },
    [handleGoogleCredential],
  );

  const logout = useCallback(async () => {
    const refreshToken = getRefreshToken();
    if (refreshToken) {
      try {
        await apiLogout(refreshToken);
      } catch {
        /* best-effort */
      }
    }
    window.google?.accounts?.id?.disableAutoSelect();
    clearSession();
    setState({ status: 'unauthenticated', user: null, error: null });
  }, []);

  return { ...state, renderGoogleButton, logout };
}
