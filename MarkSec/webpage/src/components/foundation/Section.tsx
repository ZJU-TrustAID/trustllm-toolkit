import { cn } from "@/lib/cn";

type SectionTone = "white" | "blue" | "blueStrong" | "muted";

const toneClass: Record<SectionTone, string> = {
  white: "paper-section-panel",
  blue: "paper-section-blue",
  blueStrong: "paper-section-blue-strong",
  muted: "paper-section-muted",
};

type SectionProps = {
  id?: string;
  tone?: SectionTone;
  children: React.ReactNode;
  className?: string;
};

export default function Section({ id, tone = "white", children, className }: SectionProps) {
  return (
    <section id={id} className={cn("paper-section", toneClass[tone], className)}>
      {children}
    </section>
  );
}
