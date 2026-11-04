import { cn } from "@/lib/cn";

export type PaperAuthor = {
  name: string;
  href?: string;
  orcid?: string;
  corresponding?: boolean;
};

type AuthorListProps = {
  authors: PaperAuthor[];
  className?: string;
};

export default function AuthorList({ authors, className }: AuthorListProps) {
  return (
    <p className={cn("text-base leading-7 text-[var(--muted)]", className)}>
      {authors.map((author, index) => (
        <span key={author.name}>
          {author.href || author.orcid ? (
            <a
              className="transition hover:text-[var(--ink)]"
              href={author.href ?? `https://orcid.org/${author.orcid}`}
              rel="noreferrer noopener"
              target="_blank"
            >
              {author.name}
            </a>
          ) : (
            author.name
          )}
          {author.corresponding ? <sup title="Corresponding author" aria-label="Corresponding author">*</sup> : null}
          {index < authors.length - 1 ? ", " : ""}
        </span>
      ))}
    </p>
  );
}
