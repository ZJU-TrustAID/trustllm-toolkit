import { cn } from "@/lib/cn";

type AffiliationListProps = {
  affiliations: string[];
  className?: string;
};

export default function AffiliationList({ affiliations, className }: AffiliationListProps) {
  return (
    <p className={cn("text-base leading-7 text-[var(--muted)]", className)}>
      {affiliations.join("  ")}
    </p>
  );
}
