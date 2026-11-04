import { cn } from "@/lib/cn";

type CardVariant = "default" | "flat";

const variantClass: Record<CardVariant, string> = {
  default: "border border-[var(--line)] bg-[var(--panel-solid)]",
  flat: "border border-[var(--line)] bg-[var(--panel)]",
};

type CardProps = {
  children: React.ReactNode;
  variant?: CardVariant;
  className?: string;
};

export default function Card({ children, variant = "default", className }: CardProps) {
  return <div className={cn("rounded-lg", variantClass[variant], className)}>{children}</div>;
}
