import Link from 'next/link'
import { ArrowUpRight } from 'lucide-react'
import DifficultyPill from '@/components/ui/DifficultyPill'
import StatusDot from '@/components/ui/StatusDot'
import { accentClasses, type Section } from '@/lib/roadmap'
import { cn, lessonHref } from '@/lib/utils'

interface SectionPreviewGridProps {
  section: Section
}

// Responsive 2-col card grid of lessons in a section. Used at the top of
// /learn/[section] landing pages as a more visual, airier alternative to the
// dense row-list on the home page. Each card shows the full blurb (not the
// truncated single-line version on the home roadmap), lesson index, status
// dot, and difficulty pill. The section's accent color tints the index badge
// and hover border.
export default function SectionPreviewGrid({ section }: SectionPreviewGridProps) {
  const accent = accentClasses[section.accent]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      {section.lessons.map((lesson, i) => (
        <Link
          key={lesson.slug}
          href={lessonHref(section.slug, lesson.slug)}
          className={cn(
            'group/card term-panel rounded-md p-4',
            'border-l-2 border-dark-border',
            'hover:border-dark-border-hover hover:bg-white/[0.015]',
            'transition-colors',
            'flex flex-col gap-2 min-h-[140px]'
          )}
        >
          {/* Top row: index + status + title + difficulty */}
          <div className="flex items-start gap-2.5">
            <span
              className={cn(
                'shrink-0 inline-flex items-center justify-center',
                'w-6 h-6 rounded border font-mono text-[10px] tabular-nums',
                accent.bg,
                accent.border,
                accent.text
              )}
            >
              {String(i + 1).padStart(2, '0')}
            </span>
            <div className="flex-1 min-w-0 flex items-baseline gap-2">
              <span className="flex-shrink-0 mt-[3px]">
                <StatusDot
                  sectionSlug={section.slug}
                  lessonSlug={lesson.slug}
                  interactive={false}
                />
              </span>
              <h3
                className={cn(
                  'font-mono text-[14px] text-dark-text-primary',
                  'group-hover/card:text-dark-accent transition-colors',
                  'tracking-tight leading-tight'
                )}
              >
                {lesson.title}
              </h3>
            </div>
            <DifficultyPill difficulty={lesson.difficulty} />
          </div>

          {/* Blurb */}
          <p className="pl-[34px] font-sans text-[12.5px] text-dark-text-muted leading-relaxed">
            {lesson.blurb}
          </p>

          {/* Bottom affordance */}
          <div className="pl-[34px] mt-auto flex items-center gap-1 text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled group-hover/card:text-dark-accent transition-colors">
            <span>open lesson</span>
            <ArrowUpRight
              className="w-3 h-3 group-hover/card:translate-x-0.5 group-hover/card:-translate-y-0.5 transition-transform"
              strokeWidth={1.75}
            />
          </div>
        </Link>
      ))}
    </div>
  )
}
