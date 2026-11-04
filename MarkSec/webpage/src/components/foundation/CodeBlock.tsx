import { cn } from "@/lib/cn";

type CodeBlockProps = {
  children: string;
  className?: string;
};

export default function CodeBlock({ children, className }: CodeBlockProps) {
  return (
    <pre
      className={cn(
        "overflow-x-auto rounded-md bg-[var(--terminal)] p-5 font-mono text-sm leading-7 text-white",
        className,
      )}
    >
      <code>{children}</code>
    </pre>
  );
}
