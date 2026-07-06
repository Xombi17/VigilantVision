"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, History, Settings, Users, Video, ShieldAlert } from "lucide-react";

export default function TopNav() {
  const pathname = usePathname();

  const links = [
    { href: "/", label: "Command Center", icon: LayoutDashboard },
    { href: "/cameras", label: "Surveillance", icon: Video },
    { href: "/faces", label: "Entities", icon: Users },
    { href: "/history", label: "Incident Log", icon: History },
    { href: "/settings", label: "Configuration", icon: Settings },
  ];

  return (
    <header className="glass-header sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
      <div className="flex items-center gap-8">
        <h1 className="text-2xl font-bold tracking-wider flex items-center gap-2">
          <ShieldAlert className="text-brand w-7 h-7" />
          <span><span className="text-brand">Vigilant</span>Vision</span>
        </h1>
        
        <nav className="hidden lg:flex items-center gap-2">
          {links.map((link) => {
            const Icon = link.icon;
            const isActive = pathname === link.href;

            return (
              <Link
                key={link.href}
                href={link.href}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-300 ${
                  isActive
                    ? "bg-brand/15 text-brand font-semibold shadow-[0_0_15px_rgba(20,184,166,0.1)]"
                    : "text-foreground/70 hover:bg-glass hover:text-foreground hover:scale-105"
                }`}
              >
                <Icon className="w-4 h-4" />
                <span className="text-sm">{link.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      <div className="flex items-center gap-4">
        <div className="px-3 py-1.5 rounded-full bg-danger/10 border border-danger/20 flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-danger opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-danger"></span>
          </span>
          <span className="text-xs font-semibold text-danger tracking-widest uppercase">System Armed</span>
        </div>
        <div className="w-10 h-10 rounded-full bg-glass flex items-center justify-center border border-glass-border">
          <span className="font-bold text-sm">OP</span>
        </div>
      </div>
    </header>
  );
}
