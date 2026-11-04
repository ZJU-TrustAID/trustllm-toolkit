"use client";

import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/cn";

type CopyBlockProps = {
  children: string;
  className?: string;
  codeClassName?: string;
};

export default function CopyBlock({
  children,
  className,
  codeClassName,
}: CopyBlockProps) {
  const [copied, setCopied] = useState(false);

  function handleCopy() {
    navigator.clipboard.writeText(children).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  return (
    <div className={cn("relative mt-4", className)}>
      <pre
        className={cn(
          "overflow-x-auto rounded-md border border-[var(--line)] bg-[var(--surface-muted)] p-5 font-mono text-sm leading-7 text-[var(--ink)]",
          codeClassName,
        )}
      >
        <code>{children}</code>
      </pre>
      <button
        type="button"
        onClick={handleCopy}
        className="absolute right-3 top-3 inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] bg-[var(--panel-solid)] px-3 py-1.5 font-mono text-xs text-[var(--muted)] transition hover:border-[var(--accent)] hover:text-[var(--accent-dark)]"
      >
        {copied ? (
          <>
            <Check className="h-3.5 w-3.5" />
            Copied
          </>
        ) : (
          <>
            <Copy className="h-3.5 w-3.5" />
            Copy
          </>
        )}
      </button>
    </div>
  );
}
