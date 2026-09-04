# Technical Diagram Style Specification

## 1. Visual signature

The style is a clean engineering whiteboard refined for publication: high information density, strong geometric order, restrained color, and almost no ornament.

Canonical reference: `../assets/style-reference.png`.

## 2. Palette

| Role | Suggested color | Usage |
| --- | --- | --- |
| Primary blue | `#0B4DB8` | Titles, main outlines, arrows, braces, stage numbers |
| Pale blue | `#BFD7FF` | Partition 0, selected cells, footer background |
| Teal | `#79CFC5` | Partition 1 or a second data role |
| Amber | `#F4C55E` | Partition 2, warning, intermediate state |
| Coral | `#F38A73` | Partition 3, divergent path, exception |
| Text | `#111827` | Body labels and equations |
| Grid gray | `#6B7280` | Secondary tensor grids and inactive structure |
| Canvas | `#FFFFFF` or `#F8FAFC` | Background |

Use flat fills or extremely subtle vertical shading. Maintain at least strong readable contrast. Do not use more than four semantic accent colors without a compelling reason.

## 3. Typography

- Title: bold condensed sans serif, cobalt blue, about 4–5% of canvas height.
- Stage headings: semibold blue, about 2–2.5% of canvas height.
- Labels: black or navy sans serif, short and aligned.
- Tensor shapes and equations: bold enough to survive downscaling; keep brackets and axis names exact.
- Prefer terms such as `Rank 0`, `Input [B,N,S,D]`, `local transpose`, and `AllToAll along B` over prose.

## 4. Layout grammar

```text
                    LARGE TECHNICAL TITLE

  ┌──────── Stage 1 ────────┐  ┌──────────── Stage 2 ────────────┐
  │ input → local operation │  │ ranks → communication → outputs │
  │ shape / axes / callout  │  │ stable colors and clear arrows │
  └─────────────────────────┘  └─────────────────────────────────┘

  ┌──────────────────── one-line pipeline summary ───────────────┐
  └───────────────────────────────────────────────────────────────┘
```

Preferred patterns:

- **Tensor transform:** before/after grids, axis arrows, highlighted slice, operation between them.
- **Distributed collective:** one column per rank, color-coded chunks, a central exchange network, regrouped outputs.
- **Lifecycle:** numbered horizontal stages with ownership and state changes.
- **With/without:** two aligned panels sharing the same inputs and metrics.
- **Algorithm comparison:** common pipeline first, then mutually exclusive branches, then shared output.

## 5. Shape language

- Rounded blue containers establish stage boundaries.
- Rectangular cells represent data blocks; stacked offsets imply higher-dimensional tensors.
- Braces label axes or grouped dimensions.
- Solid arrows show mandatory data flow; dashed arrows show metadata, optional flow, or references.
- Dotted vertical dividers separate compute phases without implying a data object.
- Circled numbers establish the intended reading order.

## 6. Image generation prompt template

Adapt this template rather than copying it mechanically:

```text
Create a wide 16:9 publication-quality technical infographic explaining [TOPIC].
Visual language: clean white engineering canvas, thin cobalt-blue rounded panel borders,
large centered cobalt-blue title, bold circled stage numbers, precise tensor grids and
rectangular data blocks, flat semantic fills in pale blue, teal, amber, and coral, dark
navy labels, minimal shadows, no decorative imagery.

Structure the graphic as [NUMBER] numbered stages read left to right:
1. [INPUT / BEFORE]
2. [OPERATION / COMMUNICATION]
3. [OUTPUT / AFTER]

Keep [ENTITY] the same color wherever it moves. Show exact shapes [SHAPES], axes [AXES],
and short labels [LABELS]. Put the complete transformation [SUMMARY] in a pale-blue footer
band. Avoid dark backgrounds, neon glow, 3D rendering, stock icons, long paragraphs,
ambiguous arrows, and invented technical details.
```

## 7. Technical QA

Before delivery, ask:

1. Can a reader reconstruct the transformation without the article?
2. Do input and output shapes agree with every intermediate operation?
3. Does each color preserve the same semantic identity?
4. Are local compute, communication, storage, and scheduling visually distinct?
5. Are arrow direction and ownership unambiguous?
6. Is all text readable at the final embed size?
7. Does the footer summarize rather than introduce a new concept?
