import type { ReactNode } from 'react'
import { LESSON_SECTIONS } from '../lesson-sections'
import { cn } from '@/lib/utils'

// Wraps a real lesson section in the uniform term-panel / numbered-header
// recipe that the 7-section arc uses, so anchored lessons feel consistent
// section-to-section and the TOC scrollspy can latch onto the ids.
interface LessonSectionProps {
  id: string
  children: ReactNode
}

export default function LessonSection({ id, children }: LessonSectionProps) {
  const meta = LESSON_SECTIONS.find((s) => s.id === id)
  if (!meta) {
    throw new Error(`Unknown lesson section id: ${id}`)
  }
  const index = LESSON_SECTIONS.indexOf(meta)
  const Icon = meta.icon

  return (
    <section
      id={meta.id}
      className={cn('term-panel rounded-lg overflow-hidden scroll-mt-24', 'animate-fade-up')}
      style={{ animationDelay: `${index * 40}ms` }}
    >
      <header className="term-panel-header">
        <div className="flex items-center gap-2">
          <span className="text-dark-text-disabled tabular-nums">
            {String(index + 1).padStart(2, '0')}
          </span>
          <Icon className="w-3 h-3 text-dark-accent" strokeWidth={1.75} />
          <span>{meta.label}</span>
        </div>
      </header>
      <div className="p-6 md:p-7">{children}</div>
    </section>
  )
}
