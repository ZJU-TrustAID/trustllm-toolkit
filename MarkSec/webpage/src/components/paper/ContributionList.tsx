import { cn } from "@/lib/cn";

type ContributionListProps = {
  items: string[];
  className?: string;
};

export default function ContributionList({ items, className }: ContributionListProps) {
  return (
    <div className={cn("division-panel", className)}>
      {items.map((item, index) => (
        <div className="division-row" key={item}>
          <span className="division-index">0{index + 1}</span>
          <p>{item}</p>
        </div>
      ))}
    </div>
  );
}
