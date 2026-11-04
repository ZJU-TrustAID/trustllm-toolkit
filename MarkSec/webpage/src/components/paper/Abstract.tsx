import { cn } from "@/lib/cn";

type AbstractProps = {
  children: React.ReactNode;
  className?: string;
};

export default function Abstract({ children, className }: AbstractProps) {
  return <div className={cn("prose-block", className)}>{children}</div>;
}
