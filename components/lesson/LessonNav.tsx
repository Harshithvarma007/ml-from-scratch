import Link from 'next/link'
import { ArrowLeft, ArrowRight } from 'lucide-react'
import type { Lesson } from '@/lib/roadmap'
import { roadmap } from '@/lib/roadmap'
import { cn, lessonHref } from '@/lib/utils'

interface LessonNavProps {
  prev: Lesson | null
  next: Lesson | null
}

function sectionTitle(slug: string) {
  return roadmap.find((s) => s.slug === slug)?.title ?? ''
}

export default function LessonNav({ prev, next }: LessonNavProps) {
  return (
    <nav className="mt-12 pt-6 border-t border-dark-border grid grid-cols-1 sm:grid-cols-2 gap-3">
      {prev ? (
        <Link
          href={lessonHref(prev.sectionSlug, prev.slug)}
          className={cn(
            'group term-panel rounded-md p-4 transition-all',
            'hover:border-dark-border-hover hover:bg-dark-surface-hover'
          )}
        >
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-dark-text-muted uppercase tracking-wider">
            <ArrowLeft className="w-3 h-3" />
            Previous
          </div>
          <div className="mt-1 text-[10px] font-mono text-dark-text-disabled truncate">
            {sectionTitle(prev.sectionSlug)}
          </div>
          <div className="mt-1 font-mono text-[14px] text-dark-text-primary group-hover:text-dark-accent transition-colors truncate">
            {prev.title}
          </div>
        </Link>
      ) : (
        <div className="rounded-md p-4 border border-dashed border-dark-border/40 text-[11px] font-mono text-dark-text-disabled">
          ← start of curriculum
        </div>
      )}

      {next ? (
        <Link
          href={lessonHref(next.sectionSlug, next.slug)}
          className={cn(
            'group term-panel rounded-md p-4 transition-all text-right',
            'hover:border-dark-border-hover hover:bg-dark-surface-hover'
          )}
        >
          <div className="flex items-center justify-end gap-1.5 text-[10px] font-mono text-dark-text-muted uppercase tracking-wider">
            Next
            <ArrowRight className="w-3 h-3" />
          </div>
          <div className="mt-1 text-[10px] font-mono text-dark-text-disabled truncate">
            {sectionTitle(next.sectionSlug)}
          </div>
          <div className="mt-1 font-mono text-[14px] text-dark-text-primary group-hover:text-dark-accent transition-colors truncate">
            {next.title}
          </div>
        </Link>
      ) : (
        <div className="rounded-md p-4 border border-dashed border-dark-border/40 text-[11px] font-mono text-dark-text-disabled text-right">
          end of curriculum →
        </div>
      )}
    </nav>
  )
}
