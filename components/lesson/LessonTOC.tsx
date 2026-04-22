'use client'

import { useEffect, useState } from 'react'
import { LESSON_SECTIONS } from './lesson-sections'
import { cn } from '@/lib/utils'
import Prompt from '@/components/ui/Prompt'

// Right-rail table of contents with scroll-spy.
// Tracks which lesson section is currently in view via IntersectionObserver.
export default function LessonTOC() {
  const [activeId, setActiveId] = useState<string>(LESSON_SECTIONS[0].id)

  useEffect(() => {
    const targets = LESSON_SECTIONS.map((s) => document.getElementById(s.id)).filter(
      Boolean
    ) as HTMLElement[]

    if (targets.length === 0) return

    const observer = new IntersectionObserver(
      (entries) => {
        // Pick the entry closest to the top of the viewport that's visible.
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
        if (visible.length > 0) {
          setActiveId(visible[0].target.id)
        }
      },
      { rootMargin: '-20% 0px -60% 0px', threshold: 0 }
    )

    targets.forEach((el) => observer.observe(el))
    return () => observer.disconnect()
  }, [])

  return (
    <aside className="hidden lg:block sticky top-6 w-56 flex-shrink-0">
      <div className="term-panel rounded-lg overflow-hidden">
        <header className="term-panel-header">
          <div className="flex items-center gap-2">
            <Prompt text="›" />
            <span>Contents</span>
          </div>
        </header>
        <ul className="py-2">
          {LESSON_SECTIONS.map((section, idx) => {
            const Icon = section.icon
            const active = section.id === activeId
            return (
              <li key={section.id}>
                <a
                  href={`#${section.id}`}
                  className={cn(
                    'flex items-center gap-2 px-3 py-1.5 text-left transition-all',
                    'border-l-2 font-mono text-[12px]',
                    active
                      ? 'border-dark-accent text-dark-accent bg-dark-accent/[0.06]'
                      : 'border-transparent text-dark-text-muted hover:text-dark-text-secondary hover:bg-white/[0.02]'
                  )}
                >
                  <span className="text-[10px] tabular-nums text-dark-text-disabled">
                    {String(idx + 1).padStart(2, '0')}
                  </span>
                  <Icon className="w-3 h-3 flex-shrink-0" strokeWidth={1.75} />
                  <span className="truncate">{section.shortLabel}</span>
                </a>
              </li>
            )
          })}
        </ul>
      </div>

      {/* Back-to-top */}
      <a
        href="#top"
        className={cn(
          'mt-3 inline-flex items-center gap-1.5 px-2 py-1',
          'text-[10px] font-mono text-dark-text-disabled hover:text-dark-text-secondary',
          'transition-colors'
        )}
      >
        ↑ back to top
      </a>
    </aside>
  )
}
