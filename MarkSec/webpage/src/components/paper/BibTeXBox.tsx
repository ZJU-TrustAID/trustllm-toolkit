import CodeBlock from "@/components/foundation/CodeBlock";
import { cn } from "@/lib/cn";

type BibTeXBoxProps = {
  value: string;
  className?: string;
};

export default function BibTeXBox({ value, className }: BibTeXBoxProps) {
  return (
    <div className={cn("space-y-3", className)}>
      <p className="font-mono text-xs uppercase tracking-[0.16em] text-[var(--accent-dark)]">
        BibTeX
      </p>
      <CodeBlock>{value}</CodeBlock>
    </div>
  );
}
