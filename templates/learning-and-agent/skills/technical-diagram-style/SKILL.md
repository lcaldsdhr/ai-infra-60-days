---
name: technical-diagram-style
description: Create precise, high-information technical diagrams in the repository's default blue-outline visual language. Use for AI infrastructure, distributed systems, tensor transformations, training or inference pipelines, algorithm comparisons, architecture diagrams, blog illustrations, and technical presentation graphics. Apply implicitly when a user asks to draw, visualize, illustrate, compare with/without, or generate a technical schematic unless they request another style.
---

# Technical Diagram Style

Create diagrams that teach a mechanism, not merely decorate a page. Preserve technical truth first, then optimize visual hierarchy.

Use `assets/style-reference.png` as the canonical visual reference. Read `references/style-spec.md` before producing or reviewing a diagram.

## Workflow

1. Extract the teaching claim in one sentence.
2. List entities, transformations, tensor shapes, ownership boundaries, and invariants.
3. Choose one primary reading direction: left-to-right for flow, top-to-bottom for hierarchy.
4. Split complex mechanisms into 2–6 numbered stages or clearly bounded panels.
5. Assign semantic colors once and keep them stable across every panel.
6. Add exact input/output shapes and short operation labels where they matter.
7. End with a bottom summary strip containing the complete transformation or takeaway.
8. Render and visually inspect the final artifact at full size.

## Default composition

- Use a wide 16:9 canvas with a white or very light cool-gray background.
- Place one large cobalt-blue title centered at the top.
- Use thin cobalt-blue rounded panel borders to separate conceptual stages.
- Use bold circled step numbers and short bilingual labels when established English terms aid recognition.
- Build the explanation from tensor grids, stacked blocks, rank columns, arrows, braces, axes, and concise equations.
- Reserve blue, teal, amber, and coral for distinct data partitions or roles; never use color only as decoration.
- Put the full pipeline or invariant in a pale-blue footer band.

## Tool choice

- Use `$imagegen` for polished bitmap illustrations when approximate label rendering is acceptable.
- Prefer SVG, HTML, Mermaid, PPT, or another deterministic renderer when exact Chinese text, formulas, tensor dimensions, or source-code identifiers must be flawless.
- For dense bitmap diagrams, keep generated text short. If exact wording is required, generate the graphical base first and overlay labels deterministically.
- Reuse the same style for a multi-image series: identical palette, line weight, title treatment, numbering, and footer pattern.

## Required information quality

- Show the state before, the operation, and the state after.
- Keep tensor axes, ranks, blocks, sources, and destinations explicitly distinguishable.
- Preserve entity identity through color. A blue partition must remain blue after movement or regrouping.
- Use crossing arrows only when the crossing itself teaches communication such as AllToAll.
- Label local computation separately from inter-device communication.
- State whether an illustration is conceptual or shape-accurate when simplification is unavoidable.
- Do not imply performance gains, causal relationships, or memory layouts that the source material does not support.

## Avoid

- Dark cyberpunk backgrounds, neon glow, glassmorphism, 3D chrome, decorative circuitry, and stock AI-brain imagery.
- Large paragraphs inside the image.
- More than two font families or inconsistent mathematical notation.
- Arbitrary gradients, shadows, icons, or colors without semantic meaning.
- Tiny labels, crowded legends, unexplained abbreviations, and arrows that terminate ambiguously.
- Repeating the title as a subtitle or filling empty space with ornamental elements.

## Delivery checklist

- Verify title, labels, formulas, shapes, arrows, and color identity.
- Confirm the diagram is readable at normal blog width without zooming.
- Confirm every panel contributes to the teaching claim.
- Save assets beside the consuming blog or presentation using descriptive kebab-case filenames.
- Embed the result at the first paragraph where it materially improves understanding.
