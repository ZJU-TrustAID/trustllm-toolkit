import type { ReactNode } from "react";
import Image from "next/image";
import Container from "@/components/foundation/Container";
import AffiliationList from "@/components/paper/AffiliationList";
import AuthorList, { type PaperAuthor } from "@/components/paper/AuthorList";
import ResourceButtons, { type PaperResource } from "@/components/paper/ResourceButtons";
import { cn } from "@/lib/cn";

type PaperHeroProps = {
  id?: string;
  title: ReactNode;
  subtitle?: string;
  authors?: PaperAuthor[];
  affiliations?: string[];
  contactEmails?: string[];
  resources?: PaperResource[];
  figure?: { src: string; alt: string; caption?: string };
  bordered?: boolean;
  compact?: boolean;
  balanceTitle?: boolean;
  className?: string;
};

export default function PaperHero({
  id,
  title,
  subtitle,
  authors = [],
  affiliations = [],
  contactEmails = [],
  resources = [],
  figure,
  bordered = true,
  compact = false,
  balanceTitle = true,
  className,
}: PaperHeroProps) {
  return (
    <section
      id={id}
      className={cn(
        "release-hero bg-[var(--panel-solid)]",
        bordered ? "border-b border-[var(--line)]" : null,
        className,
      )}
    >
      <Container>
        <div className="mx-auto max-w-6xl text-center">
          <h1
            className="hero-enter mx-auto max-w-5xl text-2xl font-medium leading-[1.08] tracking-normal text-[var(--ink)] sm:text-3xl md:text-4xl lg:text-5xl"
            style={{ textWrap: balanceTitle ? "balance" : "wrap" }}
          >
            {title}
          </h1>
          {subtitle ? (
            <p
              className={cn(
                "hero-enter-delay mx-auto max-w-4xl font-medium leading-snug text-[var(--muted)]",
                compact ? "mt-8 text-xl sm:text-2xl" : "mt-16 text-2xl sm:text-3xl",
              )}
            >
              {subtitle}
            </p>
          ) : null}

          {authors.length > 0 || affiliations.length > 0 ? (
            <div className="hero-enter-delay mt-8 space-y-1">
              {authors.length > 0 ? <AuthorList authors={authors} /> : null}
              {affiliations.length > 0 ? <AffiliationList affiliations={affiliations} /> : null}
              {contactEmails.length > 0 ? (
                <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 text-sm leading-6 text-[var(--muted)]">
                  {contactEmails.map((email) => <a className="break-all hover:text-[var(--ink)]" key={email} href={`mailto:${email}`}>{email}</a>)}
                </div>
              ) : null}
              {authors.some((author) => author.corresponding) ? <p className="text-sm leading-6 text-[var(--muted)]">* Corresponding author</p> : null}
            </div>
          ) : null}

          {resources.length > 0 ? (
            <ResourceButtons
              className={cn(
                "hero-enter-late",
                subtitle ? (compact ? "mt-9" : "mt-14") : "mt-10",
              )}
              resources={resources}
            />
          ) : null}

          {figure ? (
            <figure className="hero-enter-late mt-14">
              <Image
                src={figure.src}
                alt={figure.alt}
                width={1200}
                height={720}
                className="mx-auto w-full max-w-4xl rounded-lg border border-[var(--line)]"
              />
              {figure.caption ? (
                <figcaption className="mt-4 text-sm leading-6 text-[var(--muted)]">
                  {figure.caption}
                </figcaption>
              ) : null}
            </figure>
          ) : null}
        </div>
      </Container>
    </section>
  );
}
