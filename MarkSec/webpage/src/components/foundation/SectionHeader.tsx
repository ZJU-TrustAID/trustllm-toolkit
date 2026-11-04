import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/cn";

type SectionHeaderProps = {
  eyebrow?: string;
  title?: string;
  description?: string;
  icon?: LucideIcon;
  className?: string;
};

export default function SectionHeader({
  eyebrow,
  title,
  description,
  icon: Icon,
  className,
}: SectionHeaderProps) {
  return (
    <div className={cn("grid gap-6 lg:grid-cols-[0.32fr_0.68fr] lg:items-start", className)}>
      <div className="section-label">
        {Icon ? <Icon className="h-8 w-8" /> : null}
        {eyebrow}
      </div>
      <div>
        {title ? (
          <h2 className="max-w-4xl text-3xl font-semibold leading-tight text-[var(--ink)] sm:text-4xl">
            {title}
          </h2>
        ) : null}
        {description ? (
          <p className="mt-4 max-w-3xl text-base leading-7 text-[var(--muted)]">
            {description}
          </p>
        ) : null}
      </div>
    </div>
  );
}
