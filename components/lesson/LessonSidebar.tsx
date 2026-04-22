'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'
import { ChevronRight, Home } from 'lucide-react'
import SectionIcon from '@/components/ui/SectionIcon'
import StatusDot from '@/components/ui/StatusDot'
import { accentClasses, roadmap } from '@/lib/roadmap'
import { cn, lessonHref } from '@/lib/utils'

interface LessonSidebarProps {
  activeSectionSlug: string
  activeLessonSlug: string
}

export default function LessonSidebar({
  activeSectionSlug,
  activeLessonSlug,
}: LessonSidebarProps) {
  // Track which sections are expanded. Active section is always open by default.
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set([activeSectionSlug]))

  useEffect(() => {
    setExpanded((s) => new Set(s).add(activeSectionSlug))
  }, [activeSectionSlug])

  function toggle(slug: string) {
    setExpanded((s) => {
      const next = new Set(s)
      if (next.has(slug)) next.delete(slug)
      else next.add(slug)
      return next
    })
  }

  return (
    <aside className="sidebar-panel h-full w-full flex flex-col">
      {/* Header — logo + home link */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-dark-border">
        <Link href="/" className="flex items-center gap-2 group">
          <span className="font-mono text-[13px] text-dark-accent group-hover:text-dark-accent-hover transition-colors">
            ml
          </span>
          <span className="font-mono text-[13px] text-dark-text-secondary group-hover:text-dark-text-primary transition-colors">
            from·scratch
          </span>
        </Link>
        <Link
          href="/"
          title="Back to roadmap"
          className="p-1 rounded hover:bg-white/[0.05] transition-colors text-dark-text-disabled hover:text-dark-text-secondary"
        >
          <Home className="w-3 h-3" />
        </Link>
      </div>

      {/* Curriculum list */}
      <nav className="flex-1 overflow-y-auto py-2">
        {roadmap.map((section, sIdx) => {
          const isOpen = expanded.has(section.slug)
          const isActiveSection = section.slug === activeSectionSlug
          const accent = accentClasses[section.accent]

          return (
            <div key={section.slug} className="mb-1">
              {/* Section header */}
              <button
                type="button"
                onClick={() => toggle(section.slug)}
                className={cn(
                  'w-full flex items-center gap-2 px-3 py-1.5 text-left',
                  'hover:bg-white/[0.03] transition-colors group'
                )}
              >
                <ChevronRight
                  className={cn(
                    'w-3 h-3 text-dark-text-disabled transition-transform',
                    isOpen && 'rotate-90'
                  )}
                />
                <SectionIcon
                  name={section.icon}
                  className={cn('w-3 h-3', isActiveSection ? accent.text : 'text-dark-text-muted')}
                />
                <span className="text-[10px] font-mono text-dark-text-disabled tabular-nums">
                  {String(sIdx + 1).padStart(2, '0')}
                </span>
                <span
                  className={cn(
                    'flex-1 text-[11px] uppercase tracking-wider font-mono truncate',
                    isActiveSection
                      ? 'text-dark-text-primary'
                      : 'text-dark-text-secondary group-hover:text-dark-text-primary'
                  )}
                >
                  {section.title}
                </span>
              </button>

              {/* Lessons under section */}
              {isOpen && (
                <ul className="mt-0.5 mb-1 animate-fadeIn">
                  {section.lessons.map((lesson) => {
                    const active =
                      section.slug === activeSectionSlug && lesson.slug === activeLessonSlug
                    return (
                      <li key={lesson.slug}>
                        <Link
                          href={lessonHref(section.slug, lesson.slug)}
                          className={cn(
                            'flex items-center gap-2 pl-10 pr-3 py-1 rounded-sm transition-colors',
                            active
                              ? 'bg-dark-accent/10 text-dark-accent'
                              : 'text-dark-text-secondary hover:bg-white/[0.03] hover:text-dark-text-primary'
                          )}
                        >
                          <StatusDot
                            sectionSlug={section.slug}
                            lessonSlug={lesson.slug}
                          />
                          <span className="font-mono text-[12px] truncate flex-1">
                            {lesson.title}
                          </span>
                        </Link>
                      </li>
                    )
                  })}
                </ul>
              )}
            </div>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-3 py-2 border-t border-dark-border text-[10px] font-mono text-dark-text-disabled">
        <div className="flex items-center justify-between">
          <span>v1 · curriculum</span>
          <span className="tabular-nums">{roadmap.reduce((n, s) => n + s.lessons.length, 0)} lessons</span>
        </div>
      </div>
    </aside>
  )
}
