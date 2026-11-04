# Reference Sites

This document records external websites worth learning from while developing
the reusable paper-site template. Learn layout logic, pacing, hierarchy, and
interaction patterns; do not copy brand assets or proprietary content.

## Google DeepMind Genie 3

URL:
`https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/`

Use this page as a high-confidence reference for a clean research/product
announcement homepage.

What to borrow:

- Short white hero, not a full-screen hero. The next content should appear
  quickly enough that the page feels active rather than empty.
- Centered title stack with minimal metadata and very little explanatory copy.
- Large but controlled headline. Avoid forcing awkward three-line wrapping on
  tablet/mobile.
- CTA group with calm pill buttons. Buttons should scale with the headline and
  should not look heavier than the title.
- White-first visual system. Use accent color sparingly and keep the homepage
  background white for a premium feel.
- Section labels such as `Abstract`, `Method`, and `Evaluation` can be simple
  centered headings. The explanatory sentence below should stay at normal
  reading size.
- Content blocks should move up under the hero. Avoid huge blank space between
  the hero and first substantive section.
- Media grids or figure bands can create energy below the hero, but they should
  have clear rhythm and rounded corners rather than feeling like random cards.

What not to borrow blindly:

- Do not copy Google/DeepMind branding, logo behavior, copy, or assets.
- Do not add large decorative media unless the paper has meaningful figures,
  examples, traces, or qualitative outputs to show.
- Do not make every page identical to a product launch. Paper-specific method
  sections should keep their own narrative structure.

## Roman Technology Research Template

URL:
`https://research-template.roman.technology/`

Use this page as a reference for paper-site component decomposition. It is less
about a single visual mood and more about how a reusable research template can
expose common academic content blocks.

What to borrow:

- Treat the paper site as a composable kit, not one giant page component.
- Keep generic layout primitives separate from paper-specific blocks.
- Provide figure-oriented components such as wide media, comparison figures,
  carousels, and tabbed examples.
- Include academic writing utilities: LaTeX rendering, code blocks, tables,
  BibTeX, and citation sections.
- Support flexible content layouts such as single-column prose, two-column
  explanations, wide figures, and interactive comparison blocks.
- Let each example demonstrate an isolated component pattern. This helps future
  paper authors quickly see what the template can do.
- Components should feel useful even when the actual paper changes; avoid
  hard-coding a single research domain into the API.

How it informs MarkSec:

- Our first template layer should keep `foundation`, `paper`, `llm`, and
  `layout` components separate.
- `Figure`, `TableWrapper`, `BibTeXBox`, `PromptCard`, `ResponseCard`, and
  future `Tabs` / `Comparison` components are worth treating as reusable
  template primitives.
- Page-level method and case-study sections should remain flexible JSX, using
  these primitives as building blocks rather than being forced into one
  universal `MethodSection`.

What not to borrow blindly:

- Do not over-componentize every small text fragment.
- Do not force all paper sites into a demo-gallery structure if the paper needs
  a different narrative.
- Do not use template examples as production content; they should guide API and
  layout decisions only.

## Meta AI Segment Anything 2

URL:
`https://ai.meta.com/research/sam2/`

Use this page as an aesthetic reference for advanced research launch pages. It
is especially useful for studying large section blocks, strong but controlled
visual rhythm, and resource CTAs.

What to borrow:

- Clear top-down sectioning. Each major idea gets its own visual band instead
  of being hidden inside many small cards.
- Large color-block logic: white / light color / resource bands can separate
  narrative stages without heavy decoration.
- Resource links can feel premium through small interaction details, such as
  animated arrow icons inside circular outlines.
- Keep section content highly scannable: short headings, concise body copy, and
  strong figure or demo placements.
- Use a few high-value visual surfaces rather than many equal-weight cards.
- CTAs should be semantically clear: paper, code, model, demo, dataset, or
  download.

How it informs MarkSec:

- Keep the homepage white-first, then alternate white and light-blue bands for
  major sections.
- Avoid dark or fully saturated blocks unless they serve a specific resource or
  media purpose.
- Add an animated external-arrow primitive for resource links, but reserve it
  for high-value links so the page does not feel noisy.
- Prefer section-level rhythm over decorative backgrounds.

What not to borrow blindly:

- Do not copy Meta branding, iconography, copy, or assets.
- Do not overuse animated icons. Motion should indicate important resource
  actions, not decorate every link.
- Do not force media-heavy layouts when a paper is primarily text, tables, or
  traces.

## Mind2Web

URL:
`https://osu-nlp-group.github.io/Mind2Web/`

Use this page as a reference for a more academic paper site that still presents
LLM/agent research effectively. It is useful because it handles datasets,
resource links, task examples, updates, tables, an explorer, and citation in a
single long-form page.

What to borrow:

- Paper metadata is complete and immediately available: title, authors,
  affiliations, corresponding-author notes, and resource links.
- Resource buttons support more than paper/code: training data, test data, raw
  dumps, and other project artifacts.
- Task examples appear early, grounding the paper before long technical prose.
- The page supports dense academic content: abstract, introduction, data
  statistics, comparison tables, data explorer, related work, disclaimer, and
  citation.
- Agent/web-task projects benefit from explicit task descriptions and action
  traces rather than abstract claims alone.
- Dataset/project pages need update logs because resources may evolve after
  publication.

How it informs MarkSec:

- Our `ResourceButtons` type should remain open-ended enough for datasets,
  models, leaderboards, raw dumps, demos, slides, and posters.
- LLM/agent templates should include `PromptCard`, `ResponseCard`,
  `TraceViewer`, `ToolCallBlock`, `BenchmarkTable`, and `BibTeXBox` as reusable
  primitives.
- For security/watermark projects, include concrete task or attack examples
  early rather than relying only on taxonomy prose.
- Long tables and code/trace blocks must be mobile-safe through wrappers.

What not to borrow blindly:

- Do not copy the older academic visual style wholesale; keep MarkSec cleaner
  and more modern.
- Do not place every dense element above the fold. Use the structure, not the
  exact pacing.
- Do not make the page feel like a README dump; preserve designed hierarchy and
  visual breathing room.
