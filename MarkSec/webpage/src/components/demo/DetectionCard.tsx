import type { DetectionMetrics } from '@/lib/demo-types';
import { cn } from '@/lib/cn';
import { ShieldCheck, ShieldX } from 'lucide-react';

type DetectionCardProps = {
  title: string;
  detection: DetectionMetrics;
  className?: string;
};

export default function DetectionCard({ title, detection, className }: DetectionCardProps) {
  const isWatermarked = detection.is_watermarked;

  return (
    <div
      className={cn(
        'rounded-lg border p-5 transition-all',
        isWatermarked
          ? 'border-[var(--green)] bg-[var(--green-soft)]'
          : 'border-[var(--line)] bg-[var(--panel-solid)]',
        className,
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <p className="font-mono text-xs uppercase tracking-[0.16em] text-[var(--accent-dark)]">
          {title}
        </p>
        {isWatermarked ? (
          <ShieldCheck className="h-5 w-5 text-[var(--green)]" />
        ) : (
          <ShieldX className="h-5 w-5 text-[var(--muted)]" />
        )}
      </div>

      <div className="mt-3">
        <span
          className={cn(
            'text-lg font-semibold',
            isWatermarked ? 'text-[var(--green)]' : 'text-[var(--muted)]',
          )}
        >
          {detection.z_score === null ? 'Insufficient tokens' : isWatermarked ? 'Watermark Detected' : 'Below threshold'}
        </span>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3">
        <MetricItem label="Z-Score" value={detection.z_score?.toFixed(2) ?? 'N/A'} />
        <MetricItem label="Green %" value={(detection.green_fraction * 100).toFixed(1)} suffix="%" />
        <MetricItem label="Scored" value={String(detection.num_scored_tokens)} />
      </div>
      {detection.reason ? <p className="mt-3 text-sm text-[var(--muted)]">{detection.reason}</p> : null}
    </div>
  );
}

function MetricItem({
  label,
  value,
  suffix,
}: {
  label: string;
  value: string;
  suffix?: string;
}) {
  return (
    <div>
      <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-[var(--muted)]">
        {label}
      </p>
      <p className="mt-1 text-base font-semibold text-[var(--ink)]">
        {value}
        {suffix ? <span className="text-xs font-normal text-[var(--muted)]">{suffix}</span> : null}
      </p>
    </div>
  );
}
