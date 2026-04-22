import { BookOpen } from 'lucide-react'
import Link from 'next/link'
import { accentClasses, findLessonBySlug, roadmap } from '@/lib/roadmap'
import { cn, lessonHref } from '@/lib/utils'

// Top-of-lesson callout listing prerequisite lessons. Pulls from
// lib/roadmap.ts when given `currentSlug`, so the edge only has to be
// authored in one place. Empty state says "start here" so entry-point
// lessons (like gradient-descent) still get the visual frame — a reader
// landing cold knows they're not missing a lesson they should have read.
//
// Placement: above the first piece of lesson content, below the header.
// Companion to the inline <NeedsBackground> tag used mid-prose.
interface PrereqProps {
  currentSlug?: string
  slugs?: string[]
}

export default function Prereq({ currentSlug, slugs }: PrereqProps) {
  // Explicit slugs take precedence so an author can override the roadmap
  // for a one-off lesson or a narrower reading list.
  let list: string[] = []
  if (slugs && slugs.length > 0) list = slugs
  else if (currentSlug) {
    const lesson = findLessonBySlug(currentSlug)
    list = lesson?.prerequisites ?? []
  }

  const resolved = list
    .map((s) => findLessonBySlug(s))
    .filter((l): l is NonNullable<ReturnType<typeof findLessonBySlug>> => l !== null)

  const empty = resolved.length === 0

  return (
    <aside
      className={cn(
        'rounded-lg px-4 py-3 my-5 border-l-2',
        empty
          ? 'border-l-term-green/60 bg-term-green/[0.04]'
          : 'border-l-dark-accent bg-dark-accent/[0.04]',
      )}
    >
      <div className="flex items-center gap-1.5 mb-1.5">
        <BookOpen
          className={cn('w-3 h-3', empty ? 'text-term-green' : 'text-dark-accent')}
          strokeWidth={2}
        />
        <span
          className={cn(
            'text-[10px] uppercase tracking-wider font-mono',
            empty ? 'text-term-green' : 'text-dark-accent',
          )}
        >
          {empty ? 'start here' : 'before this lesson'}
        </span>
      </div>
      {empty ? (
        <p className="font-sans text-[13px] text-dark-text-secondary leading-relaxed">
          No prerequisites — this is an entry point.
        </p>
      ) : (
        <ul className="space-y-1">
          {resolved.map((l) => {
            const section = roadmap.find((s) => s.slug === l.sectionSlug)!
            const accent = accentClasses[section.accent]
            return (
              <li key={l.slug}>
                <Link
                  href={lessonHref(l.sectionSlug, l.slug)}
                  className="group flex items-center gap-2.5 py-0.5 font-sans"
                >
                  <span
                    className={cn(
                      'w-1.5 h-1.5 rounded-full flex-shrink-0',
                      accent.dot,
                    )}
                  />
                  <span className="text-[13px] text-dark-text-primary group-hover:text-dark-accent transition-colors">
                    {l.title}
                  </span>
                  <span
                    className={cn(
                      'text-[10px] font-mono uppercase tracking-wider',
                      accent.text,
                    )}
                  >
                    {section.title}
                  </span>
                </Link>
              </li>
            )
          })}
        </ul>
      )}
    </aside>
  )
}
