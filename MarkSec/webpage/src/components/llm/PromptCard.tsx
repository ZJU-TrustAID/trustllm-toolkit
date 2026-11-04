import Card from "@/components/foundation/Card";
import { cn } from "@/lib/cn";

type PromptCardProps = {
  title: string;
  label?: string;
  metadata?: string;
  children: React.ReactNode;
  className?: string;
};

export default function PromptCard({
  title,
  label = "Prompt",
  metadata,
  children,
  className,
}: PromptCardProps) {
  return (
    <Card className={cn("p-5", className)}>
      <div className="flex flex-col gap-2 border-b border-[var(--line)] pb-4 sm:flex-row sm:items-baseline sm:justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.16em] text-[var(--accent-dark)]">
            {label}
          </p>
          <h3 className="mt-2 text-lg font-semibold text-[var(--ink)]">{title}</h3>
        </div>
        {metadata ? <p className="text-sm text-[var(--muted)]">{metadata}</p> : null}
      </div>
      <div className="mt-4 text-sm leading-7 text-[var(--muted)]">{children}</div>
    </Card>
  );
}
