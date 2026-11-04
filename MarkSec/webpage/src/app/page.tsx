import {
  BookOpenText,
  FileText,
  FlaskConical,
  Layers3,
  ShieldCheck,
} from "lucide-react";
import Image from "next/image";
import Container from "@/components/foundation/Container";
import Section from "@/components/foundation/Section";
import SectionHeader from "@/components/foundation/SectionHeader";
import AbstractText from "@/components/paper/Abstract";
import PaperHero from "@/components/paper/PaperHero";
import SiteHeader from "@/components/layout/SiteHeader";
import SiteFooter from "@/components/layout/SiteFooter";
import CiteBlock from "@/components/paper/CiteBlock";
import PaperTable from "@/components/paper/PaperTable";
import { BlockMath, InlineMath } from "@/components/paper/MathFormula";
import { paper } from "@/content/paper";

const homeNavItems = [
  { href: "#abstract", label: "Abstract" },
  { href: "#taxonomy", label: "Taxonomy" },
  { href: "#marksec", label: "MarkSec" },
  { href: "#exps", label: "Exps" },
];

const heroResources = [
  {
    href: "https://arxiv.org/pdf/2609.16681",
    label: "Paper",
    svgPath:
      "M5 4h10v4h4v12H5zM3.999 2A.995.995 0 0 0 3 2.992v18.016a1 1 0 0 0 .993.992h16.014A1 1 0 0 0 21 20.992V7l-5-5zm6.5 5.5c0 1.577-.455 3.437-1.224 5.153c-.772 1.723-1.814 3.197-2.9 4.066l1.18 1.613c2.927-1.952 6.168-3.29 9.304-2.842l.457-1.939C14.644 12.661 12.5 9.99 12.5 7.5zm.6 5.972c.268-.597.505-1.216.705-1.843a9.7 9.7 0 0 0 1.706 1.966c-.982.176-1.944.465-2.875.833q.248-.471.465-.956",
  },
  {
    href: "https://github.com/ZJU-TrustAID/MarkSec",
    label: "Code",
    svgPath:
      "M5.884 18.653c-.3-.2-.558-.455-.86-.816a51 51 0 0 1-.466-.579c-.463-.575-.755-.841-1.056-.95a1 1 0 1 1 .675-1.882c.752.27 1.261.735 1.947 1.588c-.094-.117.34.427.433.539c.19.227.33.365.44.438c.204.137.588.196 1.15.14c.024-.382.094-.753.202-1.095c-2.968-.726-4.648-2.64-4.648-6.396c0-1.24.37-2.356 1.058-3.292c-.218-.894-.185-1.975.302-3.192a1 1 0 0 1 .63-.582c.081-.024.127-.035.208-.047c.803-.124 1.937.17 3.415 1.096a11.7 11.7 0 0 1 2.687-.308c.912 0 1.819.104 2.684.308c1.477-.933 2.614-1.227 3.422-1.096q.128.02.218.05a1 1 0 0 1 .616.58c.487 1.216.52 2.296.302 3.19c.691.936 1.058 2.045 1.058 3.293c0 3.757-1.674 5.665-4.642 6.392c.125.415.19.878.19 1.38c0 .665-.002 1.299-.007 2.01c0 .19-.002.394-.005.706a1 1 0 0 1-.018 1.958c-1.14.227-1.984-.532-1.984-1.525l.002-.447l.005-.705c.005-.707.008-1.337.008-1.997c0-.697-.184-1.152-.426-1.361c-.661-.57-.326-1.654.541-1.751c2.966-.333 4.336-1.482 4.336-4.66c0-.955-.312-1.744-.913-2.404A1 1 0 0 1 17.2 6.19c.166-.414.236-.957.095-1.614l-.01.003c-.491.139-1.11.44-1.858.949a1 1 0 0 1-.833.135a9.6 9.6 0 0 0-2.592-.349c-.89 0-1.772.118-2.592.35a1 1 0 0 1-.829-.134c-.753-.507-1.374-.807-1.87-.947c-.143.653-.072 1.194.093 1.607a1 1 0 0 1-.189 1.045c-.597.655-.913 1.458-.913 2.404c0 3.172 1.371 4.328 4.322 4.66c.865.097 1.202 1.177.545 1.748c-.193.168-.43.732-.43 1.364v3.15c0 .985-.834 1.725-1.96 1.528a1 1 0 0 1-.04-1.962v-.99c-.91.061-1.661-.088-2.254-.485",
  },
  {
    href: "https://arxiv.org/abs/2609.16681",
    label: "arXiv",
    svgPath:
      "M62.258 8.006a22.22 22.22 0 0 0-20.929 13.448c-3.404 8.169-.96 13.898 6.506 24.59c10.935 16.09 122.178 149.673 122.178 149.673l-24.619 23.038c-20.74 20.735-21.632 48.566-2.34 67.852l28.663 27.3l-79.976 98.235c-6.21 6.614-10.053 18.221-6.585 26.552a22.7 22.7 0 0 0 21.21 14.06a20.23 20.23 0 0 0 15.249-7.536l95.122-88.437L363.33 496.39a27.14 27.14 0 0 0 18.418 7.61a25.3 25.3 0 0 0 7.335-1.108a27.66 27.66 0 0 0 18.4-18.99a25.6 25.6 0 0 0-6.481-23.69L272.219 305.195l23.062-21.443c17.198-15.504 17.29-42.455.197-58.076l-25.257-24.228L357.417 98.46l.115-.133l.103-.14c7.793-10.123 11.52-17.92 7.502-27.806a36.17 36.17 0 0 0-23.647-18.37a24 24 0 0 0-3.166-.212l-.006.018a28.52 28.52 0 0 0-18.252 8.123l-.203.166l-.19.173L218.6 151.925L79.261 18.253S70.995 8.213 62.258 8.006m276.06 51.214q1.115.004 2.22.148a29.3 29.3 0 0 1 17.719 13.81c2.246 5.523 1.554 10.01-6.506 20.484L264.861 196.3l-40.882-39.22l100.68-91.304a21.77 21.77 0 0 1 13.66-6.536zM175.077 201.127L395.19 464.872c4.32 5.408 7.02 10.818 5.18 16.914a20.25 20.25 0 0 1-13.463 14.037a17.6 17.6 0 0 1-5.17.784a19.8 19.8 0 0 1-13.293-5.56l-220.15-209.694c-17.317-17.316-14.698-40.33 2.158-57.186z",
    svgViewBox: "0 0 448 512",
    svgWidth: "0.88em",
  },
  {
    href: "#",
    label: "Slides",
    svgPath:
      "M13 17v3h5v2H6v-2h5v-3H4a1 1 0 0 1-1-1V4H2V2h20v2h-1v12a1 1 0 0 1-1 1zm-8-2h14V4H5zm5-9l5 3.5l-5 3.5z",
  },
];

const paperAuthors = paper.authors;
const paperAffiliations = paper.affiliations;

const taxonomyBoundaries = [
  [
    "Stealing",
    "Stealing is an information-recovery objective. It does not necessarily remove a watermark by itself; instead, it recovers artifacts such as green lists, token-color maps, count tables, detector statistics, or proxy distributions that can later support stronger attacks.",
  ],
  [
    "Scrubbing",
    "Scrubbing is a removal objective. Direct scrubbers rewrite, edit, or search over a given watermarked text, while stealing-based scrubbers first recover watermark-related information and then use that artifact to guide removal.",
  ],
  [
    "Spoofing",
    "Spoofing is a forgery objective. It creates new text that is accepted as watermarked even though it was not produced by the genuine watermark process, so it targets false attribution rather than removal robustness.",
  ],
];

const experimentRows = [
  ["RQ1", "General-scrubbing robustness", "Quality filtering changes the ranking: SIRA leads raw removal, but LLMP leads quality-aware C4 scrubbing."],
  ["RQ2", "Family-specific patterns", "LLMP is the strongest quality-aware general-scrubbing baseline on C4, with narrower margins under UW and KGW."],
  ["RQ3", "Utility-aware trade-offs", "Stricter Prometheus gates preserve LLMP's lead and reduce SIRA's raw-removal advantage."],
  ["RQ4", "Instruction-tuning sensitivity", "Deployment format drives most KGW sensitivity; controlled prompting narrows the Base-Instruct gap."],
  ["RQ5", "Stealing-based attacks", "Recovered information creates scrubbing and spoofing risk, but gains depend on artifact alignment and quality gates."],
];

const resultFigures = [
  {
    src: "/marksec/paper/figures/fig3-asr-qsr.png",
    alt: "ASR-to-QSR collapse and rank shifts",
    label: "RQ1",
    title: "Overall general-scrubbing robustness",
    text: "ASR alone over-credits attacks that damage the text. Quality filtering changes both absolute success and the identity of the strongest attack.",
  },
  {
    src: "/marksec/paper/figures/fig4-qsr-heatmap.png",
    alt: "Watermark-specific QSR heatmap",
    label: "RQ2",
    title: "Family-specific robustness patterns",
    text: "The family-conditioned view shows where a broad quality-aware baseline is stable and where watermark mechanisms narrow the margin.",
  },
  {
    src: "/marksec/paper/figures/fig6-threshold.png",
    alt: "Prometheus threshold sensitivity",
    label: "RQ3",
    title: "Utility-aware robustness trade-offs",
    text: "Stricter Prometheus gates preserve LLMP's lead and reduce the apparent advantage of raw-removal-heavy attacks.",
  },
  {
    src: "/marksec/paper/figures/fig8-deployment.png",
    alt: "Deployment sensitivity under matched KGW settings",
    label: "RQ4",
    title: "Instruction-tuning sensitivity",
    text: "The controlled Base-Instruct ablation separates checkpoint effects from prompt-format effects and shows deployment format drives much of the KGW sensitivity.",
  },
  {
    src: "/marksec/paper/figures/table5-stealing.png",
    alt: "Stealing-based attacks under matched KGW settings",
    label: "RQ5",
    title: "Stealing-based attacks",
    text: "Recovered information can help in matched settings, but gains depend on artifact alignment, query/recovery assumptions, and the usable-quality gate.",
  },
];

export default function Home() {
  return (
    <main className="min-h-screen text-[var(--foreground)]">
      <Header />
      <Hero />
      <Abstract />
      <TaxonomyAtlas />
      <MarkSecSystem />
      <Experiments />
      <Citation />
      <Footer />
    </main>
  );
}

function Header() {
  return <SiteHeader navItems={homeNavItems} />;
}

function Hero() {
  return (
    <PaperHero
      id="paper-hero"
      title={
        <span className="home-hero-title">
          <span className="whitespace-nowrap">
            <span className="home-hero-marksec inline-block bg-[linear-gradient(135deg,#2b5876_0%,#536976_48%,#292e49_100%)] bg-clip-text text-transparent">
              MarkSec
            </span>
            :
          </span>
          {" "}
          Capability-Aware Evaluation of Adversarial Attacks Against LLM Watermarks
        </span>
      }
      balanceTitle={false}
      authors={paperAuthors}
      affiliations={paperAffiliations}
      contactEmails={paper.emails}
      resources={heroResources}
      bordered={false}
    />
  );
}

function Abstract() {
  return (
    <Section
      id="abstract"
      tone="white"
      className="abstract-section paper-section-no-divider"
    >
      <Container>
        <SectionHeader
          className="section-header-centered"
          icon={BookOpenText}
          eyebrow="Abstract"
        />
        <AbstractText className="mx-auto mt-8 max-w-[50rem] text-justify text-[var(--ink)]">
          {paper.abstract.map((paragraph) => (
            <p key={paragraph}>{paragraph}</p>
          ))}
        </AbstractText>
      </Container>
    </Section>
  );
}

function TaxonomyAtlas() {
  return (
    <section id="taxonomy" className="paper-section paper-section-blue">
      <div className="paper-container">
        <SectionHeader
          className="section-header-centered"
          icon={Layers3}
          eyebrow="Taxonomy"
        />

        <div className="mt-10">
          <figure className="taxonomy-paper-figure">
            <Image
              src="/marksec/paper/figures/fig1-taxonomy.png"
              alt="Attack taxonomy organized by stealing, scrubbing, and spoofing objectives"
              width={2364}
              height={1124}
            />
            <figcaption>
              Attack taxonomy organized by adversarial objective: stealing,
              scrubbing, and spoofing.
            </figcaption>
          </figure>

          <div className="taxonomy-attack-summary mx-auto max-w-3xl">
            {taxonomyBoundaries.map(([title, text], index) => (
              <div className="taxonomy-attack-item" key={title}>
                <span className="taxonomy-attack-index">{index + 1}</span>
                <div>
                  <h3>{title} attacks</h3>
                  <p>{text}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function MarkSecSystem() {
  return (
    <section id="marksec" className="paper-section paper-section-panel">
      <div className="paper-container">
        <SectionHeader
          className="section-header-centered"
          icon={ShieldCheck}
          eyebrow="MarkSec"
        />

        <figure className="hero-enter-late mx-auto mt-10 max-w-3xl">
          <Image
            src="/marksec/paper/figures/fig2-system.png"
            alt="System overview of MarkSec"
            width={1200}
            height={720}
            className="mx-auto w-full rounded-lg border border-[var(--line)]"
          />
          <figcaption className="mt-4 text-justify text-sm leading-6 text-[var(--muted)]">
            MarkSec connects stealing, scrubbing, and spoofing under explicit capability
            assumptions, shared detector calibration, and quality-constrained reporting.
          </figcaption>
        </figure>

        <div className="prose-block mx-auto mt-10 max-w-3xl text-justify text-[var(--ink)]">
          <p>{paper.abstract[1]}</p>
          <p>
            The system view above makes that framing operational: MarkSec keeps
            capability assumptions, detector calibration, and quality-aware
            reporting inside one comparable workflow.
          </p>
        </div>

        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <div className="metric-panel">
            <p className="metric-label">Robustness metric</p>
            <BlockMath>
              {"\\mathrm{ASR}=\\frac{\\mathrm{TPR}_{\\text{before}}-\\mathrm{TPR}_{\\text{after}}}{\\mathrm{TPR}_{\\text{before}}}"}
            </BlockMath>
            <p>
              ASR measures how much detector recall drops after attack, with a
              fixed detector threshold inside each benchmark cell.
            </p>
          </div>
          <div className="metric-panel metric-panel-accent">
            <p className="metric-label">Quality-constrained success</p>
            <BlockMath>
              {"\\mathrm{QSR}@\\tau=\\frac{1}{N_{\\text{all}}}\\sum_{i=1}^{N_{\\text{all}}}\\mathbf{1}[s_i=1\\wedge q_i\\ge\\tau]"}
            </BlockMath>
            <p>
              QSR counts a sample only when it removes the watermark and passes
              the utility gate. The main text uses Prometheus as{" "}
              <InlineMath>{"q_i"}</InlineMath> and reports{" "}
              <InlineMath>{"\\mathrm{QSR}@3"}</InlineMath>.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

function Experiments() {
  return (
    <section id="exps" className="paper-section paper-section-blue-strong">
      <div className="paper-container">
        <SectionHeader
          className="section-header-centered"
          icon={FlaskConical}
          eyebrow="Exps"
        />

        <div className="mt-10">
          <PaperTable
            caption="Experimental questions and headline takeaways."
            columns={["RQ", "Focus", "Takeaway"]}
            rows={experimentRows}
            note="General scrubbing RQ1-RQ3 share an input-output contract; RQ4 and RQ5 are scoped analyses with different assumptions."
          />
        </div>

        <div className="mt-8 grid gap-4 lg:grid-cols-3">
          {resultFigures.map((figure) => (
            <figure className="result-figure" key={figure.src}>
              <span>{figure.label}</span>
              <Image src={figure.src} alt={figure.alt} width={900} height={675} />
              <figcaption>
                <strong>{figure.title}</strong>
                {figure.text}
              </figcaption>
            </figure>
          ))}
        </div>
      </div>
    </section>
  );
}

function Citation() {
  return (
    <Section id="citation" tone="white">
      <Container>
        <SectionHeader className="section-header-centered" icon={FileText} eyebrow="Citation" />
        <div className="mx-auto mt-10 max-w-3xl">
          <CiteBlock />
        </div>
      </Container>
    </Section>
  );
}

function Footer() {
  return <SiteFooter note="Paper view for reading the framing, taxonomy, evidence, and citation in one pass." />;
}
