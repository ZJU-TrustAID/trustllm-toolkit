# MarkSec Webpage Design Guide

This document defines the visual and content direction for MarkSec frontend work.
Use it as the default style guide for future pages, components, and demos under
`webpage/`.

## Design Thesis

MarkSec should feel like a quiet research package console: precise, modular,
credible, and package-ready. The page should communicate that MarkSec is a
Python toolkit people can install, run, inspect, and extend, not a one-off paper
dashboard or benchmark poster.

Visual keywords:

- Research package
- Security lab
- Technical but calm
- Modular toolkit
- Reproducible workflow
- Clean release surface
- Google/DeepMind-inspired restraint

Avoid:

- Marketing-heavy SaaS landing pages
- AI dashboard spectacle
- Purple/blue gradient hero sections
- Dense benchmark matrices on the toolkit homepage
- Paper-only result framing
- Decorative card mosaics without a workflow purpose

## Current Visual Direction

The current primary direction is a Google/DeepMind-inspired research launch
style: white-first, highly restrained, centered where it matters, and light on
ornament. The page should feel premium because it removes unnecessary visual
weight, not because it adds decoration.

Core rules:

- Homepage hero is short, white, and centered. It should not occupy the full
  viewport by default.
- The first substantive content should appear soon after the hero; avoid large
  empty gaps between hero and body.
- Main title is large but controlled. Avoid awkward three-line wrapping on
  tablet and mobile.
- Hero subtitle is optional. Prefer removing it when the title and metadata are
  already clear.
- Authors and affiliations sit between the title and resource buttons, using
  normal body-sized text.
- Resource buttons are calm pills with medium weight text. They should never
  look heavier than the title.
- Section labels such as `Abstract`, `Taxonomy`, and `Evaluation` are simple,
  centered, and only slightly larger than body text.
- The explanatory title/description under a section label should use normal
  reading scale, not hero-scale type.
- Use white and light blue section bands in alternation. Do not place blue
  directly next to another blue or pale-blue band.
- Avoid dark technical blocks on the homepage unless there is a specific reason
  such as code, terminal, or resource content.
- Header uses floating, pill-like, blurred groups inspired by Google DeepMind,
  but without copying their branding.
- `marksec-icon-v1.png` already includes a circular border. Do not wrap it in
  another circular border.
- Prefer rounded-lg / 8px-ish surfaces for normal content. Avoid excessive
  `rounded-2xl` card styling unless the component genuinely needs a larger
  framed surface.
- Shadows should be rare. Prefer spacing, borders, and white/light-blue bands
  for hierarchy.

## Content Principles

Lead with toolkit affordances:

- CLI entrypoint: `marksec`
- Python package direction: `pip install marksec`
- Config-driven runs: `configs/*.yaml`
- Modular code surface: attacks, watermarks, datasets, evaluation
- Local artifacts: `output/`, SQLite registry, logs, result files
- Extension points: custom attacks, backends, loaders, metrics, schedulers

The root homepage may introduce the research framing: watermark-security
taxonomy, threat assumptions, attack families, and the paper-level motivation.
The `/package` page is the dedicated toolkit/package page.

Do not make the `/package` page primarily about built-in benchmark scope.
Benchmark matrices and paper-specific experiment coverage should live in
separate paper/docs pages. The toolkit page may mention shipped examples, but
the main story is reusable infrastructure.

Preferred `/package` page sequence:

1. Hero: product name, toolkit promise, package/terminal visual, install CTA.
2. Toolkit modules: concrete package primitives and paths.
3. Workflow: how a user moves from install to config to local artifacts.
4. Extension surface: where users can add or swap functionality.
5. Quick start: minimal commands, no long tutorial.
6. Release boundary: what belongs in the toolkit and what stays outside.

## Copy Voice

Use direct product language. Keep sentences short and technical without becoming
dry. The best copy tells users what they can do with the package.

Good:

- "A Python toolkit for building, running, and evaluating attacks against LLM watermarking systems."
- "Package parts that stay useful outside one paper."
- "Swap one layer without rewriting the stack."
- "Config files act as portable experiment specs."

Avoid:

- Broad claims such as "next-generation AI security platform".
- Result claims that belong to a paper.
- Long explanations in the UI.
- Repeating "benchmark" so often that MarkSec sounds like a fixed matrix.

## Layout System

Use a restrained, wide research-site layout:

- Max content width: `max-w-7xl`.
- Horizontal padding: `px-5` on mobile, `lg:px-8` on desktop.
- Section rhythm: generous vertical padding, typically `py-20` to `py-24`.
- Header height: `h-16`, sticky, translucent, thin bottom border.
- Hero height: at least `calc(100svh - 4rem)` when using the sticky header.
- Prefer two-column sections for narrative and implementation details.
- Use cards only for module items, command surfaces, repeated lists, or framed tooling.
- Do not nest cards inside cards.

Hero composition:

- Brand/product name must be the largest text.
- The hero visual should be a package console, module map, or code/runtime surface.
- Do not use a generic abstract SVG illustration.
- Do not make the hero a centered text-only block.

## Color Tokens

Themes live in `src/app/globals.css`. The default theme is `blueprint`, enabled
from `src/app/layout.tsx`:

```tsx
<html lang="en" data-theme="blueprint">
```

Available theme values:

- `blueprint`: default, clean technical blue with `#0058d0` as the primary color.
- `scholar`: brighter paper-like surface with the same primary blue.
- `lab`: cool teal support colors around the same primary blue.
- `night`: dark technical mode for comparison.

Default palette:

```css
:root {
  --background: #f4f8ff;
  --foreground: #111827;
  --ink: #0b1220;
  --muted: #506070;
  --panel-solid: #ffffff;
  --surface-muted: #eaf2ff;
  --accent: #0058d0;
  --accent-dark: #003f99;
  --green: #087f5b;
  --amber: #b45309;
}
```

Usage:

- `--background`: main page field.
- `--ink`: headings, primary actions, strong labels.
- `--muted`: body text and secondary navigation.
- `--line`: default section and panel borders.
- `--accent`: primary action and technical emphasis.
- `--surface-muted` / `--surface-soft`: section bands and small code labels.
- `--header-bg`: translucent sticky header background.
- `--green`: success, available extension points, positive checks.
- `--amber`: caution or neutral status, used sparingly.

Keep `#0058d0` as the main brand blue, but avoid making the whole page one-note:
pair it with clean white surfaces, graphite text, restrained green success
signals, and warm amber only where a status needs it.

## Background And Surfaces

The page background is intentionally plain. Use section dividers and soft bands
for hierarchy instead of a global grid or decorative background fields:

```css
body {
  background: var(--background);
}
```

Surface classes:

- `.lab-panel`: glassy tool surfaces with thin border, blur, and soft shadow.
- `.lab-panel-flat`: flatter repeated items with border and translucent fill.
- `.terminal-grid`: dark console surface with a subtle grid.

Use shadows sparingly. If removing shadows breaks the visual hierarchy, improve
spacing, border, or contrast first.

## Typography

Use the current type system:

```css
--font-sans-stack: "Segoe UI", "Helvetica Neue", "PingFang SC", "Noto Sans SC", sans-serif;
--font-serif-stack: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", "Noto Serif SC", serif;
--font-mono-stack: "IBM Plex Mono", "SFMono-Regular", "Consolas", monospace;
```

Rules:

- Product name and main section titles use sans, large and confident.
- A small italic serif accent may be used for a secondary hero word such as "Toolkit".
- Use mono for package paths, commands, runtime labels, and small technical tags.
- Do not use negative letter spacing.
- Use uppercase tracking only for small kicker labels and mono tags.

Approximate scale:

- Hero product name: `text-6xl` to `lg:text-8xl`.
- Hero secondary word: `text-4xl` to `lg:text-6xl`.
- Section titles: `text-4xl` to `md:text-5xl`.
- Body: `text-base` or `text-lg` with relaxed line height.
- Module text: `text-sm` with `leading-6`.

## Components

Header:

- Sticky, translucent, blurred.
- Left: icon mark plus mono `MARKSEC`.
- Center/right: 3 to 4 anchor links.
- Right CTA: dark or accent button, short label.

Section kicker:

- Small rounded pill.
- Mono uppercase text.
- One lucide icon.
- Border and translucent panel fill.

Buttons:

- Primary: accent background, white text, small icon.
- Secondary: transparent/off-white background, border, dark text.
- Radius: `rounded-md`.
- Hover: color or tiny icon translate, not dramatic movement.

Module tiles:

- Use a shared grid with one-pixel dividers rather than many floating cards.
- Include one icon, one module name, one path/code label, and one short sentence.
- Hover may brighten the tile and invert the icon container.

Terminal/code surfaces:

- Dark graphite background: `#101820`.
- Mono text, compact line height.
- Use low-contrast grid lines and a subtle scan line.
- Include real commands and real module names.

## Motion

Motion should create hierarchy and presence, not spectacle.

Current motion primitives:

- `heroEnter`: staggered hero entrance.
- `slowDrift`: subtle floating movement for the console visual.
- `scanLine`: slow scan over the terminal surface.
- Button icon hover: tiny horizontal movement.

Rules:

- Keep durations calm: roughly 700ms for entrance, several seconds for ambient motion.
- Always support `prefers-reduced-motion`.
- Do not animate large sections in a way that blocks reading.
- Motion should be removable without changing layout.

## Iconography

Use `lucide-react` icons. Pick icons for scanning, not decoration.

Good mappings:

- `Terminal`: CLI, package console, commands.
- `FileCode2`: config/spec files.
- `Wrench`: implementation modules.
- `ShieldCheck`: watermark/security adapters.
- `Database`: datasets and local registry.
- `FlaskConical`: evaluation utilities.
- `GitBranch`: extension surface.
- `CircuitBoard`: MarkSec identity mark.

Avoid custom SVG icons unless MarkSec gets a formal logo system.

## Technical Stack

Match the current reference stack:

- Next.js App Router
- React
- TypeScript
- Tailwind CSS v4
- `lucide-react`
- CSS variables in `globals.css`

Keep pages mostly static and fast. Use client components only when interaction
actually requires browser state.

## Future Page Guidance

For package/docs pages:

- Keep the same header, palette, and section rhythm.
- Use real commands and file paths as visual anchors.
- Prefer docs-like clarity over campaign copy.

For demos:

- Start with the working surface, not a marketing hero.
- Preserve the same restrained colors and mono labels.
- Show status, inputs, outputs, and artifacts clearly.

For paper or benchmark pages:

- It is acceptable to include matrices, result tables, and paper framing there.
- Keep them visually related to the toolkit site, but do not move that content
  back into the toolkit homepage.
