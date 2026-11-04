import { cn } from "@/lib/cn";

export type PaperResource = {
  label: string;
  href: string;
  svgPath?: string;
  svgViewBox?: string;
  svgWidth?: string;
  svgHeight?: string;
};

type ResourceButtonsProps = {
  resources: PaperResource[];
  className?: string;
};

export default function ResourceButtons({ resources, className }: ResourceButtonsProps) {
  return (
    <div className={cn("flex flex-row flex-wrap justify-center gap-3", className)}>
      {resources.map((resource) => (
        <a
          key={`${resource.label}-${resource.href}`}
          href={resource.href || undefined}
          aria-disabled={resource.href ? undefined : true}
          title={resource.href ? undefined : "Link forthcoming"}
          className="flex flex-row items-center gap-2 rounded-full bg-[var(--ink)] px-5 py-2 text-lg font-medium text-white hover:bg-[var(--accent-dark)] hover:no-underline aria-disabled:cursor-default aria-disabled:opacity-60"
        >
          {resource.svgPath ? (
            <svg
              width={resource.svgWidth ?? "1em"}
              height={resource.svgHeight ?? "1em"}
              viewBox={resource.svgViewBox ?? "0 0 24 24"}
            >
              <path fill="currentColor" d={resource.svgPath} />
            </svg>
          ) : null}
          <span>{resource.label}</span>
        </a>
      ))}
    </div>
  );
}
