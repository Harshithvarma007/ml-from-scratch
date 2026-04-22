import Link from 'next/link'
import type { ReactNode } from 'react'
import { findLessonBySlug } from '@/lib/roadmap'
import { cn, lessonHref } from '@/lib/utils'

// Inline "you should've read X first" tag. Renders like a Wikipedia inline
// reference — the prose reads naturally and the reader gets a quiet click
// target if the term is unfamiliar. Use on the FIRST mention of a concept
// borrowed from an earlier lesson. Later mentions stay plain.
//
//   <NeedsBackground slug="recurrent-neural-network">RNNs</NeedsBackground>
interface NeedsBackgroundProps {
  slug: string
  children: ReactNode
}

export default function NeedsBackground({ slug, children }: NeedsBackgroundProps) {
  const lesson = findLessonBySlug(slug)
  if (!lesson) {
    // Unknown slug — render plain children so the page still works, but
    // loudly log the typo in dev. Silent in production; surfaces won't
    // break because the component degrades to an identity wrapper.
    if (process.env.NODE_ENV !== 'production') {
      console.warn(`[NeedsBackground] unknown lesson slug: "${slug}"`)
    }
    return <>{children}</>
  }
  return (
    <Link
      href={lessonHref(lesson.sectionSlug, lesson.slug)}
      title={`Background: ${lesson.title}`}
      className={cn(
        'text-dark-text-primary',
        'border-b border-dashed border-dark-accent/40 hover:border-dark-accent',
        'hover:text-dark-accent transition-colors',
      )}
    >
      {children}
    </Link>
  )
}
