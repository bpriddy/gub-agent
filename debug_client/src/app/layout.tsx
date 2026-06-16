import type { Metadata } from 'next';
import Script from 'next/script';
import './globals.css';

export const metadata: Metadata = {
  title: 'gub-agent debug',
  description: 'Local debug client for gub-agent decomposition traces',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {/* Google Identity Services — loaded before interactive so the
            sign-in button can render. */}
        <Script src="https://accounts.google.com/gsi/client" strategy="beforeInteractive" />
        {children}
      </body>
    </html>
  );
}
