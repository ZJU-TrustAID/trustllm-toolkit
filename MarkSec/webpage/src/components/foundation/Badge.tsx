import { cn } from "@/lib/cn";

type BadgeProps = {
  children: React.ReactNode;
  className?: string;
};

export default function Badge({ children, className }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border border-[var(--line)] bg-[var(--panel)] px-3 py-1 font-mono text-xs uppercase tracking-[0.16em] text-[var(--accent-dark)]",
        className,
      )}
    >
      {children}
    </span>
  );
}
