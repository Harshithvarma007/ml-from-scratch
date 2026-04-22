# Contributing to ML from Scratch

Thanks for wanting to contribute. This doc covers everything you need: the lesson writing style, how to add a widget, and the PR process. Read it once; it's not long.

---

## Table of Contents

- [What we want](#what-we-want)
- [What we don't want](#what-we-dont-want)
- [Project setup](#project-setup)
- [How to add a lesson](#how-to-add-a-lesson)
- [Lesson writing style](#lesson-writing-style)
- [How to add a widget](#how-to-add-a-widget)
- [PR checklist](#pr-checklist)
- [Good first issues](#good-first-issues)

---

## What we want

- **New lessons** — especially gaps in the current roadmap (e.g., layer norm from scratch, speculative decoding deep-dive)
- **Widget improvements** — better mobile responsiveness, accessibility (focus rings, aria labels), cleaner animations
- **Derivation fixes** — if a math step is wrong, missing, or glossed over, open an issue or PR
- **Typos and clarity** — fix anything that reads awkwardly; the house voice is dry but precise
- **`KeyTerm` additions** — wrapping the first occurrence of defined terms in `<KeyTerm>` across lessons
- **References** — adding papers/posts/code links to the `<References>` block at the end of a lesson

## What we don't want

- Framework swaps (React, Next.js, Tailwind are fixed)
- Dark/light mode toggle — dark only, intentional
- Changing a lesson's algorithm or data to produce different outputs — content correctness is the author's call
- Persistence / accounts / backend — the site is intentionally local-state only
- Lessons that skip the derivation and jump straight to PyTorch — the three-layer pattern is load-bearing

---

## Project setup

```bash
git clone https://github.com/YOUR_USERNAME/ml-from-scratch.git
cd ml-from-scratch
npm install
npm run dev          # http://localhost:3000
npx tsc --noEmit    # must stay at zero errors
```

The only required check before a PR is `npx tsc --noEmit`. CI runs it on every push.

---

## How to add a lesson

There are exactly three steps.

### Step 1 — Add the lesson entry to `lib/roadmap.ts`

Find the correct section's `lessons: [...]` array and add:

```ts
{
  slug: 'layer-normalization',          // kebab-case, URL-safe
  title: 'Layer Normalization',         // display name
  difficulty: 'Medium',                 // 'Easy' | 'Medium' | 'Hard'
  blurb: 'One sentence — what this is and why it matters.',
  prerequisites: ['batch-normalization'],   // optional: slugs of lessons this depends on
  enables: ['transformer-block'],           // optional: slugs this unlocks
},
```

Insert it in the reading order you'd recommend to a learner. The position in the array is the position in the section landing page and sidebar.

### Step 2 — Create the lesson file

Create `components/lesson/content/lessons/layer-normalization.tsx`:

```tsx
import MathBlock from '../MathBlock'
import {
  Prose,
  Callout,
  Bridge,
  Gotcha,
  Challenge,
  KeyTerm,
  Eq,
  EqRef,
} from '../primitives'

export default function LayerNormalizationLesson() {
  return (
    <div className="space-y-6">
      <Prose>
        <p>
          Start with the intuition. What problem does this solve? Why does
          batch norm fail for sequences and small batches?
        </p>
      </Prose>

      <Eq id="layer-norm" number="1.1" caption="layer normalization">
{`y = (x − μ) / √(σ² + ε)  ·  γ  +  β`}
      </Eq>

      {/* ... rest of lesson */}
    </div>
  )
}
```

See [Lesson writing style](#lesson-writing-style) below for the full voice guide.

### Step 3 — Register the component

Open `components/lesson/content/registry.ts` and add two lines:

```ts
// At the top with other imports:
import LayerNormalizationLesson from './lessons/layer-normalization'

// In the REGISTRY object, in roadmap order:
'pytorch/layer-normalization': { Component: LayerNormalizationLesson },
```

The key is `'section-slug/lesson-slug'` — both slugs must match exactly what's in `lib/roadmap.ts`.

### Verify

```bash
npx tsc --noEmit   # zero errors
npm run dev        # navigate to /learn/[section]/layer-normalization
```

---

## Lesson writing style

This is the most important section. Lessons have a specific voice and structure. Violating it makes the contribution feel out of place.

### Voice

- **Dry, precise, no filler.** "This is important" is banned. Show why it's important.
- **Speak to one reader.** "You" not "we" or "the student". First person is fine for the author voice.
- **Humor earns its place.** A wry aside is welcome; a forced joke is not. When in doubt, cut it.
- **Short paragraphs.** Two to four sentences. If a paragraph runs longer, split it.
- **Math is mandatory, not decorative.** Every claim that has a derivation gets the derivation. Skipping steps is fine; hiding that you skipped is not.

### Structure

Every lesson should follow this order loosely:

1. **Opening hook** — the problem this solves. One or two paragraphs, no math yet.
2. **The core idea** — intuition first, then the equation. Use `<Eq>` for numbered equations.
3. **The three-layer code pattern** — see below.
4. **A `<Bridge>`** — a table connecting concepts the reader already knows to the new ones.
5. **A `<Callout>`** — one key insight, danger, or misconception. Not more than one per lesson.
6. **The widget** — the interactive moment. Place it after the theory, not before.
7. **`<Gotcha>` blocks** — practical failure modes, in order of likelihood.
8. **`<Challenge>`** — a concrete extension exercise. Should be do-able in 15 minutes.
9. **Takeaway + next up** — two short paragraphs. What to carry forward. What comes next.
10. **`<References>`** — papers, blog posts, course notes. URLs required.

Not every lesson needs every section. All lessons need 1, 2, 6, 9, and 10.

### Three-layer code pattern

Every algorithm gets three implementations in a `<LayeredCode>` block:

```tsx
import LayeredCode from '../LayeredCode'
import CodeBlock from '../CodeBlock'

<LayeredCode
  tabs={['Pure Python', 'NumPy', 'PyTorch']}
  defaultTab="NumPy"
>
  <CodeBlock lang="python" filename="layer_norm_pure.py">
    {`# Explicit loops — maximum clarity, minimum speed`}
  </CodeBlock>
  <CodeBlock lang="python" filename="layer_norm_numpy.py">
    {`# NumPy vectorised — the "from scratch" canonical version`}
  </CodeBlock>
  <CodeBlock lang="python" filename="layer_norm_torch.py">
    {`# PyTorch one-liner — show the shortcut after earning it`}
  </CodeBlock>
</LayeredCode>
```

The Pure Python tab shows every loop explicitly. The NumPy tab is the "from scratch" canonical version — this is what the lesson derives. The PyTorch tab shows the library shortcut and notes what it does differently (fused kernels, autocast, etc.).

### Primitives reference

| Component | Use for |
|-----------|---------|
| `<Prose>` | All body text. Wraps `<p>` tags with correct spacing. |
| `<Callout variant="insight">` | One key insight per lesson. Variants: `insight`, `warning`, `danger`. |
| `<Bridge rows={[...]}>` | Two-column table: left = familiar concept, right = new concept. |
| `<Gotcha>` | Practical failure modes. Can contain multiple `<p>` children. |
| `<Challenge prompt="title">` | Extension exercise. Add a runnable `<CodeBlock>` inside. |
| `<Personify speaker="X">` | Give a concept a first-person voice. Used sparingly. |
| `<KeyTerm>term</KeyTerm>` | Wrap the **first** occurrence of each defined term. |
| `<Eq id="..." number="1.1">` | Numbered equation with anchor. `id` must be globally unique. |
| `<EqRef id="..." number="1.1" />` | Cross-reference an equation. Renders as a clickable `(1.1)`. |
| `<MathBlock caption="...">` | Un-numbered display math. Use `Eq` for anything you'll reference. |

---

## How to add a widget

Widgets live in `components/lesson/content/widgets/`. Every widget:

1. Is a React client component (`'use client'` at top if it needs state/effects)
2. Wraps itself in `<WidgetFrame label="Widget Title" caption="Optional one-liner">` from `./WidgetFrame`
3. Is responsive: works at 320 px wide minimum
4. Has visible focus rings on every interactive element (`focus-visible:ring-2 focus-visible:ring-dark-accent`)

### Static widget (SVG / canvas)

```tsx
'use client'
import WidgetFrame from './WidgetFrame'

export default function MyWidget() {
  return (
    <WidgetFrame label="My Widget" caption="What the user can explore here.">
      {/* SVG, canvas, or div-based visualization */}
    </WidgetFrame>
  )
}
```

### Runnable Python widget (Pyodide)

Use the existing `RunnableCodeBlock` component for code-editor-style widgets:

```tsx
import CodeBlock from '../CodeBlock'

<CodeBlock lang="python" filename="example.py" runnable>
  {`import numpy as np
print(np.array([1, 2, 3]) ** 2)`}
</CodeBlock>
```

For custom widgets that need to run arbitrary Python and parse the output (like the gradient checker), import directly from `lib/pyodide`:

```ts
import { getPyodide, ensurePackagesForSource, resetPythonGlobals } from '@/lib/pyodide'
```

Pyodide is loaded lazily on first use from the jsdelivr CDN. The first run takes ~3 seconds; subsequent runs are fast. Always show a loading indicator.

---

## PR checklist

Before opening a PR, confirm:

- [ ] `npx tsc --noEmit` — zero TypeScript errors
- [ ] New lesson: entry in `lib/roadmap.ts`, component file, registry entry — all three present
- [ ] New widget: responsive at 320 px, focus rings on interactive elements
- [ ] Lesson follows the three-layer code pattern for any algorithm implementation
- [ ] `<References>` block present with at least one URL
- [ ] No `console.log` left in production code
- [ ] PR description explains *what* and *why*, not just *what*

---

## Good first issues

Look for the [`good first issue`](https://github.com/YOUR_USERNAME/ml-from-scratch/labels/good%20first%20issue) label. Typical examples:

- Add `<KeyTerm>` wrapping to a lesson that's missing it
- Add a missing `<References>` block to a lesson
- Fix a mobile layout issue in a specific widget (search for `grid-cols-[` with fixed pixel values)
- Improve the `aria-label` on a slider or button in a widget
- Fix a typo or improve a derivation explanation

If you want to add a full lesson or widget, open a Discussion first so we can align on scope before you write the code.

---

Questions? Open a [Discussion](https://github.com/YOUR_USERNAME/ml-from-scratch/discussions). Bug? Open an [Issue](https://github.com/YOUR_USERNAME/ml-from-scratch/issues).
