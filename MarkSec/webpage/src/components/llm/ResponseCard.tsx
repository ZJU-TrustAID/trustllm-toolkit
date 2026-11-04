import Card from "@/components/foundation/Card";
import { cn } from "@/lib/cn";

type ResponseStatus = "correct" | "incorrect" | "partial" | "neutral";

const statusClass: Record<ResponseStatus, string> = {
  correct: "text-[var(--green)]",
  incorrect: "text-[var(--amber)]",
  partial: "text-[var(--accent)]",
  neutral: "text-[var(--muted)]",
};

type ResponseCardProps = {
  model: string;
  status?: ResponseStatus;
  score?: string;
  children: React.ReactNode;
  className?: string;
};

export default function ResponseCard({
  model,
  status = "neutral",
  score,
  children,
  className,
}: ResponseCardProps) {
  return (
    <Card className={cn("p-5", className)}>
      <div className="flex flex-col gap-2 border-b border-[var(--line)] pb-4 sm:flex-row sm:items-baseline sm:justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.16em] text-[var(--accent-dark)]">
            Response
          </p>
          <h3 className="mt-2 text-lg font-semibold text-[var(--ink)]">{model}</h3>
        </div>
        <div className="flex gap-3 text-sm font-semibold">
          <span className={statusClass[status]}>{status}</span>
          {score ? <span className="text-[var(--muted)]">{score}</span> : null}
        </div>
      </div>
      <div className="mt-4 text-sm leading-7 text-[var(--muted)]">{children}</div>
    </Card>
  );
}
