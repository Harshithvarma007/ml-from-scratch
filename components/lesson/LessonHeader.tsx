import Link from 'next/link'
import { Clock } from 'lucide-react'
import DifficultyPill from '@/components/ui/DifficultyPill'
import Prompt from '@/components/ui/Prompt'
import SectionIcon from '@/components/ui/SectionIcon'
import { accentClasses, type Lesson, type Section } from '@/lib/roadmap'
import { cn } from '@/lib/utils'

interface LessonHeaderProps {
  section: Section
  lesson: Lesson
}

// Simple heuristic for read time — each stub is ~15 min once filled in.
const PLACEHOLDER_READ_MIN = 15

export default function LessonHeader({ section, lesson }: LessonHeaderProps) {
  const accent = accentClasses[section.accent]

  return (
    <header className="mb-8 animate-fade-up">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-2 text-[11px] font-mono text-dark-text-muted">
        <Prompt />
        <Link
          href="/"
          className="hover:text-dark-text-secondary transition-colors uppercase tracking-wider"
        >
          curriculum
        </Link>
        <span className="text-dark-text-disabled">/</span>
        <Link
          href={`/#${section.slug}`}
          className={cn(
            'flex items-center gap-1 hover:text-dark-text-secondary transition-colors uppercase tracking-wider',
            accent.text
          )}
        >
          <SectionIcon name={section.icon} className="w-3 h-3" />
          {section.title}
        </Link>
        <span className="text-dark-text-disabled">/</span>
        <span className="text-dark-text-secondary uppercase tracking-wider truncate">
          {lesson.title}
        </span>
      </nav>

      {/* Title */}
      <h1 className="mt-5 font-mono text-3xl md:text-4xl text-dark-text-primary leading-tight tracking-tight">
        {lesson.title}
      </h1>

      {/* Blurb */}
      <p className="mt-3 font-sans text-[15px] text-dark-text-secondary leading-relaxed max-w-2xl">
        {lesson.blurb}
      </p>

      {/* Meta row */}
      <div className="mt-5 flex flex-wrap items-center gap-3">
        <DifficultyPill difficulty={lesson.difficulty} />
        <div className="flex items-center gap-1.5 text-[11px] font-mono text-dark-text-muted">
          <Clock className="w-3 h-3" />
          <span>~{PLACEHOLDER_READ_MIN} min read</span>
        </div>
        <span className="text-dark-text-disabled">·</span>
        <span className="text-[11px] font-mono text-dark-text-muted uppercase tracking-wider">
          lesson {section.lessons.findIndex((l) => l.slug === lesson.slug) + 1} of{' '}
          {section.lessons.length}
        </span>
      </div>
    </header>
  )
}
