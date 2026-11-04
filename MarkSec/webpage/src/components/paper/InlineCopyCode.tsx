"use client";

import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/cn";

type InlineCopyCodeProps = {
  value: string;
  className?: string;
};

export default function InlineCopyCode({
  value,
  className,
}: InlineCopyCodeProps) {
  const [copied, setCopied] = useState(false);

  function handleCopy() {
    navigator.clipboard.writeText(value).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    });
  }

  return (
    <button
      type="button"
      onClick={handleCopy}
      className={cn("inline-copy-code", className)}
      aria-label={copied ? "Copied command" : "Copy command"}
    >
      <code>{value}</code>
      <span className="inline-copy-code-indicator" aria-hidden="true">
        {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      </span>
    </button>
  );
}
