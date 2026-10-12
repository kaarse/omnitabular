import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OmniTabular | Advanced Tabular ML Framework",
  description: "The Ultimate Tabular ML Framework with 32+ Architectures and Auto Ensemble Optimization.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
        <footer className="footer glass">
          <p>Built with ❤️ and Math implicitly in PyTorch. © {new Date().getFullYear()} OmniTabular ML</p>
        </footer>
      </body>
    </html>
  );
}
