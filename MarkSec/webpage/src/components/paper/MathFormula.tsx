import katex from "katex";
import { cn } from "@/lib/cn";

type MathFormulaProps = {
  children: string;
  className?: string;
};

function renderMath(value: string, displayMode: boolean) {
  return {
    __html: katex.renderToString(value, {
      displayMode,
      throwOnError: false,
      strict: "ignore",
    }),
  };
}

export function InlineMath({ children, className }: MathFormulaProps) {
  return (
    <span
      className={cn("math-inline", className)}
      dangerouslySetInnerHTML={renderMath(children, false)}
    />
  );
}

export function BlockMath({ children, className }: MathFormulaProps) {
  return (
    <div
      className={cn("math-block", className)}
      dangerouslySetInnerHTML={renderMath(children, true)}
    />
  );
}
