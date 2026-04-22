## What does this PR do?

<!-- One paragraph. What changed and why. Not a list of files. -->

## Type of change

- [ ] New lesson
- [ ] New widget
- [ ] Bug fix
- [ ] Widget improvement (responsiveness, accessibility, polish)
- [ ] Lesson content fix (typo, derivation, clarity)
- [ ] Infrastructure / tooling

## For new lessons

- [ ] Entry added to `lib/roadmap.ts` (slug, title, difficulty, blurb)
- [ ] Component file at `components/lesson/content/lessons/[slug].tsx`
- [ ] Registered in `components/lesson/content/registry.ts`
- [ ] Uses `<LayeredCode>` for algorithm implementations (Pure Python / NumPy / PyTorch)
- [ ] Has a `<References>` block with at least one URL
- [ ] `<Challenge>` block present with a concrete extension exercise

## For new widgets

- [ ] Responsive at 320 px viewport width
- [ ] Focus rings on all interactive elements (`focus-visible:ring-2 focus-visible:ring-dark-accent`)
- [ ] `aria-label` on sliders and icon-only buttons
- [ ] No hardcoded pixel widths in SVG without `preserveAspectRatio`

## Checklist (all PRs)

- [ ] `npx tsc --noEmit` — zero errors
- [ ] Tested in Chrome and Firefox
- [ ] Mobile layout checked (375 px)
- [ ] No `console.log` left in
- [ ] Linked to related issue (if any): closes #

## Screenshots / demo

<!-- For UI changes: before/after screenshots or a short screen recording. -->
