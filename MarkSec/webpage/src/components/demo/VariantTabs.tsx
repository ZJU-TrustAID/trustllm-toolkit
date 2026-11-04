import type { DemoVariant } from '@/lib/demo-types';
import { cn } from '@/lib/cn';

type VariantTabsProps = {
  variants: DemoVariant[];
  activeIndex: number;
  onSelect: (index: number) => void;
};

export default function VariantTabs({ variants, activeIndex, onSelect }: VariantTabsProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {variants.map((variant, index) => {
        const active = index === activeIndex;
        return (
          <button
            key={variant.label}
            type="button"
            onClick={() => onSelect(index)}
            className={cn(
              'min-w-[140px] rounded-lg border px-4 py-3 text-left transition-all',
              active
                ? 'border-[var(--accent)] bg-[var(--accent-soft)]'
                : 'border-[var(--line)] bg-[var(--panel-solid)] hover:border-[var(--line-strong)]',
            )}
          >
            <div
              className={cn(
                'text-xs font-semibold uppercase tracking-[0.18em]',
                active ? 'text-[var(--accent-dark)]' : 'text-[var(--muted)]',
              )}
            >
              {variant.label}
            </div>
            <div className="mt-1 text-sm text-[var(--ink)]">
              z = {variant.detection.z_score?.toFixed(2) ?? 'N/A'}
            </div>
          </button>
        );
      })}
    </div>
  );
}
