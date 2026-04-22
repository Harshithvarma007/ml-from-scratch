import Link from 'next/link'
import { ArrowLeft, ArrowRight, ArrowUpLeft } from 'lucide-react'
import SectionIcon from '@/components/ui/SectionIcon'
import Prompt from '@/components/ui/Prompt'
import SectionPreviewGrid from './SectionPreviewGrid'
import { accentClasses, roadmap, type Section } from '@/lib/roadmap'
import { cn, sectionHref } from '@/lib/utils'

interface SectionLandingProps {
  section: Section
}

// Full-page landing for /learn/[section]. Mirrors the terminal-panel aesthetic
// of the home page but at a single-section granularity: hero with accent +
// icon + difficulty histogram, then a visual 2-col SectionPreviewGrid of
// every lesson (vs. the dense row-list on the home roadmap), then prev/next
// section pagination. Intentionally skips the full-site sidebar — this page
// is an overview, not a lesson shell.
export default function SectionLanding({ section }: SectionLandingProps) {
  const accent = accentClasses[section.accent]
  const index = roadmap.findIndex((s) => s.slug === section.slug)
  const prev = index > 0 ? roadmap[index - 1] : null
  const next = index < roadmap.length - 1 ? roadmap[index + 1] : null

  const counts = {
    Easy: section.lessons.filter((l) => l.difficulty === 'Easy').length,
    Medium: section.lessons.filter((l) => l.difficulty === 'Medium').length,
    Hard: section.lessons.filter((l) => l.difficulty === 'Hard').length,
  }

  return (
    <div className="min-h-screen bg-dark-bg">
      <div className="max-w-5xl mx-auto px-6 lg:px-8 pt-10 pb-16">
        {/* Back crumb */}
        <Link
          href="/#curriculum"
          className={cn(
            'inline-flex items-center gap-1.5 mb-6',
            'text-[11px] font-mono text-dark-text-muted uppercase tracking-wider',
            'hover:text-dark-accent transition-colors'
          )}
        >
          <ArrowUpLeft className="w-3 h-3" strokeWidth={2} />
          roadmap
        </Link>

        {/* Hero */}
        <header
          className={cn(
            'term-panel rounded-lg overflow-hidden',
            'border-l-2 mb-10 animate-fade-up',
            accent.border
          )}
        >
          <div className="p-6 md:p-8">
            <div className="flex items-start gap-4">
              {/* Icon badge */}
              <div
                className={cn(
                  'shrink-0 flex items-center justify-center',
                  'w-12 h-12 rounded border',
                  accent.bg,
                  accent.border,
                  accent.text,
                  accent.glow
                )}
              >
                <SectionIcon name={section.icon} className="w-5 h-5" />
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 text-[11px] font-mono text-dark-text-muted uppercase tracking-wider">
                  <Prompt />
                  section {String(index + 1).padStart(2, '0')} of{' '}
                  {String(roadmap.length).padStart(2, '0')}
                </div>
                <h1 className="mt-2 font-mono text-2xl md:text-3xl text-dark-text-primary tracking-tight">
                  {section.title}
                </h1>
                <p className="mt-2 font-sans text-[14px] text-dark-text-secondary leading-relaxed max-w-2xl">
                  {section.subtitle}
                </p>
              </div>
            </div>

            {/* Difficulty histogram */}
            <div className="mt-6 flex flex-wrap items-center gap-2 text-[10px] font-mono uppercase tracking-wider">
              <span className="text-dark-text-disabled">
                {section.lessons.length}{' '}
                {section.lessons.length === 1 ? 'lesson' : 'lessons'}
              </span>
              <span className="text-dark-text-disabled">·</span>
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
          </div>
        </header>

        {/* Lessons grid */}
        <div className="mb-12">
          <div className="mb-4 flex items-baseline justify-between">
            <h2 className="font-mono text-[13px] text-dark-text-muted uppercase tracking-wider">
              Lessons
            </h2>
            <span className="text-[10px] font-mono text-dark-text-disabled tabular-nums">
              in order
            </span>
          </div>
          <SectionPreviewGrid section={section} />
        </div>

        {/* Prev / Next section nav */}
        {(prev || next) && (
          <nav
            aria-label="Section navigation"
            className="grid grid-cols-1 md:grid-cols-2 gap-3"
          >
            {prev ? (
              <Link
                href={sectionHref(prev.slug)}
                className={cn(
                  'group/nav term-panel rounded-md p-4',
                  'border-l-2 border-dark-border hover:border-dark-border-hover',
                  'hover:bg-white/[0.015] transition-colors',
                  'flex items-center gap-3'
                )}
              >
                <ArrowLeft
                  className="w-4 h-4 text-dark-text-disabled group-hover/nav:text-dark-accent group-hover/nav:-translate-x-0.5 transition-all"
                  strokeWidth={1.75}
                />
                <div className="min-w-0">
                  <div className="text-[10px] font-mono text-dark-text-disabled uppercase tracking-wider">
                    previous section
                  </div>
                  <div className="font-mono text-[13px] text-dark-text-primary group-hover/nav:text-dark-accent transition-colors truncate">
                    {prev.title}
                  </div>
                </div>
              </Link>
            ) : (
              <div />
            )}

            {next ? (
              <Link
                href={sectionHref(next.slug)}
                className={cn(
                  'group/nav term-panel rounded-md p-4',
                  'border-l-2 border-dark-border hover:border-dark-border-hover',
                  'hover:bg-white/[0.015] transition-colors',
                  'flex items-center justify-end gap-3 md:text-right'
                )}
              >
                <div className="min-w-0">
                  <div className="text-[10px] font-mono text-dark-text-disabled uppercase tracking-wider">
                    next section
                  </div>
                  <div className="font-mono text-[13px] text-dark-text-primary group-hover/nav:text-dark-accent transition-colors truncate">
                    {next.title}
                  </div>
                </div>
                <ArrowRight
                  className="w-4 h-4 text-dark-text-disabled group-hover/nav:text-dark-accent group-hover/nav:translate-x-0.5 transition-all"
                  strokeWidth={1.75}
                />
              </Link>
            ) : (
              <div />
            )}
          </nav>
        )}
      </div>
    </div>
  )
}
