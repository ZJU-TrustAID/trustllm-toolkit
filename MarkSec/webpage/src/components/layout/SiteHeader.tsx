"use client";

import Image from "next/image";
import Link from "next/link";
import { PackageCheck, PanelsTopLeft } from "lucide-react";
import { usePathname } from "next/navigation";

type NavItem = {
  href: string;
  label: string;
};

type PageLink = {
  href: string;
  label: string;
  icon: "demo" | "toolkit";
};

type SiteHeaderProps = {
  navItems: NavItem[];
  className?: string;
};

const pageLinkIcon = {
  demo: PanelsTopLeft,
  toolkit: PackageCheck,
} as const;

const pageLinks: PageLink[] = [
  { href: "/demo", label: "Demo", icon: "demo" },
  { href: "/toolkit", label: "Toolkit", icon: "toolkit" },
];

export default function SiteHeader({
  navItems,
}: SiteHeaderProps) {
  const pathname = usePathname();

  return (
    <header id="top" className="site-header">
      <div className="site-header-blur" aria-hidden="true">
        <span className="site-header-blur-layer site-header-blur-layer-1" />
        <span className="site-header-blur-layer site-header-blur-layer-2" />
        <span className="site-header-blur-layer site-header-blur-layer-3" />
      </div>
      <div className="site-header-inner">
        <Link href="/#top" className="site-header-brand" aria-label="MarkSec home">
          <span className="site-header-logo">
            <Image
              src="/marksec/marksec-icon-transparent.png"
              alt=""
              width={44}
              height={44}
              className="h-11 w-11 object-contain"
              priority
            />
          </span>
          <span className="site-header-wordmark">MarkSec</span>
        </Link>

        <nav className="site-header-nav" aria-label="Main navigation">
          {navItems.map((item) => (
            <a className="site-header-nav-link" href={item.href} key={item.href}>
              {item.label}
            </a>
          ))}
        </nav>

        <div className="site-header-actions">
          {pageLinks.map((item) => {
            const Icon = pageLinkIcon[item.icon];
            const isActive = pathname === item.href;
            if (isActive) {
              return (
                <span
                  className="site-header-page-link is-active"
                  aria-current="page"
                  key={item.href}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.label}</span>
                </span>
              );
            }
            return (
              <Link href={item.href} className="site-header-page-link" key={item.href}>
                <Icon className="h-4 w-4" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </div>
      </div>
    </header>
  );
}
