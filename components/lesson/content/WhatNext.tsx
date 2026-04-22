import Link from 'next/link'
import { ArrowLeft, ArrowRight, ExternalLink, Sparkles } from 'lucide-react'
import type { ReactNode } from 'react'
import { findLessonBySlug, getEnabledBy, type Lesson } from '@/lib/roadmap'
import { lessonHref, cn } from '@/lib/utils'

interface DeeperLink {
  title: string
  url: string
  /** Optional one-line blurb. */
  note?: string
}

interface WhatNextProps {
  /**
   * Slug of the current lesson. We use this to compute the "builds on this"
   * (enables) side of the graph via the inverse index in roadmap.ts.
   */
  currentSlug: string
  /**
   * Curated prerequisite slugs. Falls back to the Lesson&apos;s own
   * `prerequisites` field if omitted — handy for one-off overrides.
   */
  prerequisites?: string[]
  /**
   * Curated downstream-lesson slugs. Overrides the auto-computed inverse
   * index; pass this when you want "read next" to be a specific short list
   * instead of "every lesson that lists me as a prereq."
   */
  enables?: string[]
  /** Optional external follow-ups — papers, blog posts, reference impls. */
  deeper?: DeeperLink[]
  /**
   * Hide the prerequisites card even if the data is available. Useful on
   * very first lessons in a section.
   */
  hidePrerequisites?: boolean
}

// End-of-lesson "where to next" card grid. Three buckets: what this lesson
// builds on, what builds on it, and optional external follow-ups.
//
// Designed to gracefully degrade — if no data for a bucket, we just omit the
// card, so the component is safe to drop into any lesson even before the
// graph is fully authored.
export default function WhatNext({
  currentSlug,
  prerequisites,
  enables,
  deeper,
  hidePrerequisites,
}: WhatNextProps) {
  const self = findLessonBySlug(currentSlug)

  const prereqSlugs = prerequisites ?? self?.prerequisites ?? []
  const enablesSlugs = enables ?? self?.enables

  const prereqLessons = prereqSlugs
    .map((s) => findLessonBySlug(s))
    .filter((l): l is Lesson => l !== null)

  // If the author didn't hand-curate `enables`, derive from inverse index.
  const enablesLessons = enablesSlugs
    ? enablesSlugs
        .map((s) => findLessonBySlug(s))
        .filter((l): l is Lesson => l !== null)
    : getEnabledBy(currentSlug)

  const hasPrereq = !hidePrerequisites && prereqLessons.length > 0
  const hasEnables = enablesLessons.length > 0
  const hasDeeper = deeper && deeper.length > 0

  // Nothing to show — render nothing instead of an empty box.
  if (!hasPrereq && !hasEnables && !hasDeeper) return null

  const cardCount = Number(hasPrereq) + Number(hasEnables) + Number(hasDeeper)
  const gridClass =
    cardCount === 1
      ? 'grid-cols-1'
      : cardCount === 2
        ? 'grid-cols-1 md:grid-cols-2'
        : 'grid-cols-1 md:grid-cols-3'

  return (
    <section
      aria-label="What to read next"
      className={cn('term-panel rounded-lg overflow-hidden my-6')}
    >
      <header className="term-panel-header">
        <div className="flex items-center gap-2">
          <ArrowRight className="w-3 h-3 text-dark-accent" strokeWidth={2} />
          <span>what next</span>
        </div>
      </header>
      <div className={cn('grid gap-3 p-4', gridClass)}>
        {hasPrereq && (
          <WhatNextCard
            title="Builds on"
            icon={<ArrowLeft className="w-3 h-3" strokeWidth={2} />}
            accent="text-term-cyan"
            hint="If a step here felt fast, revisit these first."
          >
            <ul className="space-y-1.5">
              {prereqLessons.map((lesson) => (
                <li key={lesson.slug}>
                  <Link
                    href={lessonHref(lesson.sectionSlug, lesson.slug)}
                    className="group flex items-start gap-2 font-sans text-[13px] text-dark-text-primary hover:text-dark-accent transition-colors"
                  >
                    <span className="text-dark-text-disabled group-hover:text-dark-accent pt-0.5">
                      ‹
                    </span>
                    <span className="flex-1">
                      <span className="underline decoration-dark-border group-hover:decoration-dark-accent underline-offset-2">
                        {lesson.title}
                      </span>
                      <span className="block text-[11.5px] text-dark-text-muted leading-snug mt-0.5">
                        {lesson.blurb}
                      </span>
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          </WhatNextCard>
        )}

        {hasEnables && (
          <WhatNextCard
            title="Unlocks"
            icon={<ArrowRight className="w-3 h-3" strokeWidth={2} />}
            accent="text-term-green"
            hint="Natural continuations that build directly on this."
          >
            <ul className="space-y-1.5">
              {enablesLessons.map((lesson) => (
                <li key={lesson.slug}>
                  <Link
                    href={lessonHref(lesson.sectionSlug, lesson.slug)}
                    className="group flex items-start gap-2 font-sans text-[13px] text-dark-text-primary hover:text-dark-accent transition-colors"
                  >
                    <span className="flex-1">
                      <span className="underline decoration-dark-border group-hover:decoration-dark-accent underline-offset-2">
                        {lesson.title}
                      </span>
                      <span className="block text-[11.5px] text-dark-text-muted leading-snug mt-0.5">
                        {lesson.blurb}
                      </span>
                    </span>
                    <span className="text-dark-text-disabled group-hover:text-dark-accent pt-0.5">
                      ›
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          </WhatNextCard>
        )}

        {hasDeeper && (
          <WhatNextCard
            title="Go deeper"
            icon={<Sparkles className="w-3 h-3" strokeWidth={2} />}
            accent="text-term-purple"
            hint="Papers, implementations, and write-ups worth your time."
          >
            <ul className="space-y-1.5">
              {deeper!.map((link) => (
                <li key={link.url}>
                  <a
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group flex items-start gap-2 font-sans text-[13px] text-dark-text-primary hover:text-dark-accent transition-colors"
                  >
                    <span className="flex-1">
                      <span className="underline decoration-dark-border group-hover:decoration-dark-accent underline-offset-2">
                        {link.title}
                      </span>
                      {link.note && (
                        <span className="block text-[11.5px] text-dark-text-muted leading-snug mt-0.5">
                          {link.note}
                        </span>
                      )}
                    </span>
                    <ExternalLink className="w-3 h-3 text-dark-text-disabled group-hover:text-dark-accent shrink-0 mt-0.5" />
                  </a>
                </li>
              ))}
            </ul>
          </WhatNextCard>
        )}
      </div>
    </section>
  )
}

interface WhatNextCardProps {
  title: string
  icon: ReactNode
  accent: string
  hint: string
  children: ReactNode
}

function WhatNextCard({ title, icon, accent, hint, children }: WhatNextCardProps) {
  return (
    <div className="rounded-md border border-dark-border bg-dark-surface/60 p-3 flex flex-col gap-2">
      <div className="flex items-center gap-1.5">
        <span className={accent}>{icon}</span>
        <span
          className={cn(
            'text-[10px] uppercase tracking-wider font-mono',
            accent,
          )}
        >
          {title}
        </span>
      </div>
      <p className="text-[11px] text-dark-text-muted leading-snug">{hint}</p>
      <div className="mt-0.5">{children}</div>
    </div>
  )
}
