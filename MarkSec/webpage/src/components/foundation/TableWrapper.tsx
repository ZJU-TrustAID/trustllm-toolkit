import { cn } from "@/lib/cn";

type TableWrapperProps = {
  children: React.ReactNode;
  className?: string;
};

export default function TableWrapper({ children, className }: TableWrapperProps) {
  return (
    <div className={cn("overflow-x-auto rounded-lg border border-[var(--line)]", className)}>
      {children}
    </div>
  );
}
