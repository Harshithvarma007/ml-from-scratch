import {
  Lightbulb,
  Sigma,
  MousePointer2,
  Code2,
  Flame,
  Microscope,
  BookMarked,
  type LucideIcon,
} from 'lucide-react'

// The 7-section arc that an anchored-TOC lesson follows. Narrative-mode
// lessons (hideTOC: true in content/registry.ts) ignore these anchors.
// Consumers: LessonTOC (scrollspy), content/LessonSection (section wrapper).
export interface LessonSectionMeta {
  id: string
  label: string
  shortLabel: string
  icon: LucideIcon
  description: string
}

export const LESSON_SECTIONS: LessonSectionMeta[] = [
  {
    id: 'intuition',
    label: 'Intuition',
    shortLabel: 'Intuition',
    icon: Lightbulb,
    description:
      'Plain-English walkthrough. No math, no code — just the mental model and why it matters.',
  },
  {
    id: 'math',
    label: 'Mathematical Foundation',
    shortLabel: 'Math',
    icon: Sigma,
    description:
      'Every equation derived step by step, with notation explained and no skipped lines.',
  },
  {
    id: 'lab',
    label: 'Interactive Lab',
    shortLabel: 'Lab',
    icon: MousePointer2,
    description:
      'Drag sliders, change hyperparameters, watch the algorithm update live. Build intuition by poking at it.',
  },
  {
    id: 'scratch',
    label: 'From Scratch (NumPy)',
    shortLabel: 'NumPy',
    icon: Code2,
    description:
      'A minimal, heavily commented reference implementation — every line earns its keep.',
  },
  {
    id: 'pytorch',
    label: 'PyTorch Equivalent',
    shortLabel: 'PyTorch',
    icon: Flame,
    description:
      'Side-by-side with the library version. See what PyTorch abstracts away and how.',
  },
  {
    id: 'deep-dive',
    label: 'Deep Dive',
    shortLabel: 'Deep Dive',
    icon: Microscope,
    description:
      'Research-level nuances: edge cases, proofs, failure modes, optimizations, open questions.',
  },
  {
    id: 'papers',
    label: 'Papers & Next',
    shortLabel: 'Papers',
    icon: BookMarked,
    description: 'Seminal papers, modern variants, and the next lessons that build on this one.',
  },
]
