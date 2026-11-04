import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/cn";

type ButtonVariant = "primary" | "secondary" | "ghost";
type ButtonSize = "md" | "lg";

const variantClass: Record<ButtonVariant, string> = {
  primary: "bg-[var(--ink)] text-white hover:bg-[var(--accent-dark)]",
  secondary:
    "bg-[var(--surface-muted)] text-[var(--ink)] hover:bg-[var(--accent-soft)] hover:text-[var(--accent-dark)]",
  ghost: "text-[var(--ink)] hover:text-[var(--accent-dark)]",
};

const sizeClass: Record<ButtonSize, string> = {
  md: "h-11 px-5 text-sm",
  lg: "h-12 px-6 text-base",
};

type ButtonProps = {
  href: string;
  children: React.ReactNode;
  variant?: ButtonVariant;
  size?: ButtonSize;
  icon?: LucideIcon;
  className?: string;
};

export default function Button({
  href,
  children,
  variant = "primary",
  size = "md",
  icon: Icon,
  className,
}: ButtonProps) {
  return (
    <a
      href={href}
      className={cn(
        "inline-flex items-center justify-center gap-2.5 rounded-full font-medium transition",
        variantClass[variant],
        sizeClass[size],
        className,
      )}
    >
      {Icon ? <Icon className={cn(size === "lg" ? "h-5 w-5" : "h-4 w-4")} /> : null}
      {children}
    </a>
  );
}
