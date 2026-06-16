/**
 * next.config.mjs
 *
 * Rewrites /gub/* → the GUB backend so the browser-side auth calls
 * (sign-in, token refresh) are same-origin and avoid CORS. The debug
 * client's auth lib points BASE_URL at `/gub`.
 *
 * GUB_BACKEND_URL is server-side (read at config load). Defaults to the
 * local GUB on :3000.
 */
const GUB_BACKEND_URL = process.env.GUB_BACKEND_URL ?? 'http://localhost:3000';

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      { source: '/gub/:path*', destination: `${GUB_BACKEND_URL}/:path*` },
    ];
  },
};

export default nextConfig;
