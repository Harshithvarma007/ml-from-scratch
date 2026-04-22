import Link from 'next/link'
import { ArrowUpRight } from 'lucide-react'
import SectionIcon from '@/components/ui/SectionIcon'
import { accentClasses, type Section } from '@/lib/roadmap'
import { cn, sectionHref } from '@/lib/utils'

interface SectionCardProps {
  section: Section
  index: number
}

// Home-page topic card — one of 14 in a 2-col grid. Replaces the previous
// RoadmapSection that listed every lesson inline on the home page. The home
// page is now a curriculum map; drilling into a section (→ /learn/[section])
// reveals the full lesson grid with blurbs and progress dots. Each card
// intentionally stays compact: icon badge, index, title, subtitle, a
// difficulty histogram, and a preview of the first two lesson titles so the
// card has enough texture to signal what's inside without becoming a list.
export default function SectionCard({ section, index }: SectionCardProps) {
  const accent = accentClasses[section.accent]
  const previewLessons = section.lessons.slice(0, 2)
  const remaining = section.lessons.length - previewLessons.length

  const counts = {
    Easy: section.lessons.filter((l) => l.difficulty === 'Easy').length,
    Medium: section.lessons.filter((l) => l.difficulty === 'Medium').length,
    Hard: section.lessons.filter((l) => l.difficulty === 'Hard').length,
  }

  return (
    <Link
      id={section.slug}
      href={sectionHref(section.slug)}
      className={cn(
        'group/card term-panel rounded-lg overflow-hidden',
        'border-l-2 transition-colors',
        accent.border,
        'hover:border-dark-border-hover hover:bg-white/[0.015]',
        'flex flex-col'
      )}
    >
      {/* Header row: icon + index + title + chevron */}
      <div className="flex items-start gap-3 px-5 pt-5">
        <div
          className={cn(
            'shrink-0 flex items-center justify-center',
            'w-9 h-9 rounded border',
            accent.bg,
            accent.border,
            accent.text
          )}
        >
          <SectionIcon name={section.icon} className="w-4 h-4" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-2 text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            <span className="tabular-nums">
              {String(index + 1).padStart(2, '0')} / {String(14).padStart(2, '0')}
            </span>
            <span>·</span>
            <span>
              {section.lessons.length}{' '}
              {section.lessons.length === 1 ? 'lesson' : 'lessons'}
            </span>
          </div>
          <h3
            className={cn(
              'mt-1 font-mono text-[16px] text-dark-text-primary tracking-tight',
              'group-hover/card:text-dark-accent transition-colors'
            )}
          >
            {section.title}
          </h3>
        </div>
        <ArrowUpRight
          className={cn(
            'shrink-0 mt-1 w-3.5 h-3.5 text-dark-text-disabled',
            'group-hover/card:text-dark-accent',
            'group-hover/card:translate-x-0.5 group-hover/card:-translate-y-0.5',
            'transition-all'
          )}
          strokeWidth={1.75}
        />
      </div>

      {/* Subtitle */}
      <p className="px-5 mt-2.5 font-sans text-[12.5px] text-dark-text-secondary leading-relaxed">
        {section.subtitle}
      </p>

      {/* Difficulty histogram */}
      <div className="px-5 mt-3 flex flex-wrap items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider">
        {counts.Easy > 0 && (
          <span className="inline-flex items-center gap-1 px-1.5 py-[2px] rounded-sm border border-term-green/30 bg-term-green/5 text-term-green">
            <span className="tabular-nums">{counts.Easy}</span>
            <span>easy</span>
          </span>
        )}
        {counts.Medium > 0 && (
          <span className="inline-flex items-center gap-1 px-1.5 py-[2px] rounded-sm border border-term-amber/30 bg-term-amber/5 text-term-amber">
            <span className="tabular-nums">{counts.Medium}</span>
            <span>medium</span>
          </span>
        )}
        {counts.Hard > 0 && (
          <span className="inline-flex items-center gap-1 px-1.5 py-[2px] rounded-sm border border-term-pink/30 bg-term-pink/5 text-term-pink">
            <span className="tabular-nums">{counts.Hard}</span>
            <span>hard</span>
          </span>
        )}
      </div>

      {/* Lesson preview bullets */}
      <ul className="px-5 mt-4 space-y-1">
        {previewLessons.map((lesson) => (
          <li
            key={lesson.slug}
            className="flex items-center gap-2 font-mono text-[12px] text-dark-text-muted"
          >
            <span
              className={cn(
                'inline-block w-1 h-1 rounded-full shrink-0',
                accent.dot,
                'opacity-50'
              )}
            />
            <span className="truncate">{lesson.title}</span>
          </li>
        ))}
        {remaining > 0 && (
          <li className="pl-3 font-mono text-[11px] text-dark-text-disabled tabular-nums">
            + {remaining} more
          </li>
        )}
      </ul>

      {/* Bottom affordance */}
      <div
        className={cn(
          'mt-4 px-5 py-3 flex items-center gap-1',
          'border-t border-dark-border/40',
          'text-[10px] font-mono uppercase tracking-wider',
          'text-dark-text-disabled group-hover/card:text-dark-accent transition-colors'
        )}
      >
        <span>enter section</span>
        <ArrowUpRight
          className="w-3 h-3 group-hover/card:translate-x-0.5 group-hover/card:-translate-y-0.5 transition-transform"
          strokeWidth={1.75}
        />
      </div>
    </Link>
  )
}
