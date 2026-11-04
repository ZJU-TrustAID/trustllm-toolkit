'use client';

import type { TokenAnnotation } from '@/lib/demo-types';
import { cn } from '@/lib/cn';

type TokenTextProps = {
  tokens: TokenAnnotation[];
  coloring: boolean;
  className?: string;
  verbatim?: boolean;
};

export default function TokenText({ tokens, coloring, className, verbatim = false }: TokenTextProps) {
  return (
    <div className={cn('whitespace-pre-wrap break-words text-[15px] leading-7', className)}>
      {tokens.map((token, index) => (
        <TokenSpan
          key={index}
          token={token}
          coloring={coloring}
          needsLeadingSpace={!verbatim && shouldInsertSpace(tokens[index - 1], token)}
        />
      ))}
    </div>
  );
}

const NO_LEADING_SPACE_TOKENS = new Set(['.', ',', ':', '-']);
const NO_TRAILING_SPACE_TOKENS = new Set(['-']);

function shouldInsertSpace(previousToken: TokenAnnotation | undefined, token: TokenAnnotation) {
  if (!previousToken) {
    return false;
  }

  if (NO_LEADING_SPACE_TOKENS.has(token.text)) {
    return false;
  }

  if (NO_TRAILING_SPACE_TOKENS.has(previousToken.text)) {
    return false;
  }

  return true;
}

function TokenSpan({
  token,
  coloring,
  needsLeadingSpace,
}: {
  token: TokenAnnotation;
  coloring: boolean;
  needsLeadingSpace: boolean;
}) {
  const colorClass = coloring
    ? {
        green: 'bg-[rgba(8,127,91,0.14)] text-[var(--green)]',
        red: 'bg-[rgba(220,38,38,0.12)] text-[#dc2626]',
        neutral: 'text-[var(--ink)]',
      }[token.label]
    : 'text-[var(--ink)]';

  return (
    <span
      className={cn('px-[0.06rem] py-[0.02rem] transition-colors duration-200 rounded-sm', colorClass)}
      title={
        token.position !== undefined
          ? `Pos: ${token.position} | ${token.label}`
          : token.label
      }
    >
      {needsLeadingSpace ? ' ' : ''}
      {token.text}
    </span>
  );
}
