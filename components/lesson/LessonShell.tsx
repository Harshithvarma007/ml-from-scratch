import type { Lesson, Section } from '@/lib/roadmap'
import { getAdjacent } from '@/lib/roadmap'
import LessonSidebar from './LessonSidebar'
import LessonHeader from './LessonHeader'
import LessonTOC from './LessonTOC'
import LessonNav from './LessonNav'
import { CompletionToggle } from './CompletionToggle'
import { getLessonContent } from './content/registry'
import ReadingProgress from './ReadingProgress'
import LessonKeyboardNav from './LessonKeyboardNav'
import MobileNav from './MobileNav'
import { cn } from '@/lib/utils'

interface LessonShellProps {
  section: Section
  lesson: Lesson
}

// 3-column desktop layout:  [sidebar w-64] [center max-w-3xl] [TOC w-56]
// Collapses to single-column below md with a hamburger-triggered drawer so the
// full curriculum is still one tap away on mobile. A thin accent bar at the
// top tracks reading progress, and ← / → jump between adjacent lessons.
// Narrative-mode lessons hide the TOC and widen the content column so widgets
// can fill the available space.
export default function LessonShell({ section, lesson }: LessonShellProps) {
  const { prev, next } = getAdjacent(section.slug, lesson.slug)
  const content = getLessonContent(section.slug, lesson.slug)
  const Content = content?.Component
  const hideTOC = content?.hideTOC ?? false

  return (
    <div id="top" className="min-h-screen bg-dark-bg flex">
      {/* Interactive islands */}
      <ReadingProgress />
      <LessonKeyboardNav prev={prev} next={next} />
      <MobileNav activeSectionSlug={section.slug} activeLessonSlug={lesson.slug} />

      {/* Left rail (desktop) */}
      <div className="hidden md:block w-64 flex-shrink-0 border-r border-dark-border sticky top-0 h-screen">
        <LessonSidebar
          activeSectionSlug={section.slug}
          activeLessonSlug={lesson.slug}
        />
      </div>

      {/* Main column — extra top padding on mobile so hamburger doesn't overlap */}
      <main className="flex-1 min-w-0 px-6 lg:px-10 py-8 pt-16 md:pt-8">
        <div className="flex gap-10 max-w-[1200px] mx-auto">
          <article
            className={cn(
              'flex-1 min-w-0',
              hideTOC ? 'max-w-4xl mx-auto' : 'max-w-3xl'
            )}
          >
            <LessonHeader section={section} lesson={lesson} />
            <CompletionToggle sectionSlug={section.slug} lessonSlug={lesson.slug} />
            {Content ? (
              <Content />
            ) : (
              <div className="term-panel rounded-lg p-8 text-center font-mono text-[13px] text-dark-text-disabled">
                Lesson content is not yet registered. Check back soon.
              </div>
            )}
            <LessonNav prev={prev} next={next} />
          </article>

          {!hideTOC && <LessonTOC />}
        </div>
      </main>
    </div>
  )
}
