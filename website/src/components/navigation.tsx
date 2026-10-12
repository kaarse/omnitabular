"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function Navigation() {
  const pathname = usePathname();
  return (
    <nav className="navbar glass" aria-label="Main navigation">
      <Link className="logo" href="/">OmniTabular</Link>
      <ul className="nav-links">
        {[["/", "Home"], ["/dashboard", "Dashboard"], ["/api-reference", "Documentation"]].map(([href, label]) => (
          <li key={href}><Link href={href} aria-current={pathname === href ? "page" : undefined}>{label}</Link></li>
        ))}
        <li><a href="https://github.com/kaarse/omnitabular" target="_blank" rel="noreferrer">GitHub ↗</a></li>
      </ul>
    </nav>
  );
}
