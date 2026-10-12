import type { Metadata } from "next";
import Link from "next/link";
import { Navigation } from "@/components/navigation";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "OmniTabular | Tabular machine learning", template: "%s | OmniTabular" },
  description: "Train, compare, and combine 32 tabular machine learning models with cross-validation and optimized ensemble weights.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <a className="skip-link" href="#main-content">Skip to content</a>
        <div className="blob-bg" aria-hidden="true"><div className="blob blob-1" /><div className="blob blob-2" /></div>
        <Navigation />
        <main id="main-content" tabIndex={-1}>{children}</main>
        <footer className="footer glass">
          <p>OmniTabular · Open source tabular machine learning</p>
          <p><Link href="/api-reference">Documentation</Link> · <a href="https://github.com/kaarse/omnitabular/issues">Report an issue</a></p>
        </footer>
      </body>
    </html>
  );
}
