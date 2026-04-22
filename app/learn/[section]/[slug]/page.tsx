import { notFound } from 'next/navigation'
import type { Metadata } from 'next'
import LessonShell from '@/components/lesson/LessonShell'
import { flatLessons, getLesson, getSection } from '@/lib/roadmap'

interface PageProps {
  params: { section: string; slug: string }
}

// Pre-render every lesson route at build time.
export function generateStaticParams() {
  return flatLessons().map((l) => ({ section: l.sectionSlug, slug: l.slug }))
}

export function generateMetadata({ params }: PageProps): Metadata {
  const lesson = getLesson(params.section, params.slug)
  const section = getSection(params.section)
  if (!lesson || !section) return { title: 'Lesson not found' }
  const title = lesson.title
  return {
    title,
    description: lesson.blurb,
    openGraph: {
      title: `${lesson.title} · ML from Scratch`,
      description: lesson.blurb,
      type: 'article',
      url: `/learn/${section.slug}/${lesson.slug}`,
    },
    twitter: {
      card: 'summary_large_image',
      title: `${lesson.title} · ML from Scratch`,
      description: lesson.blurb,
    },
    alternates: {
      canonical: `/learn/${section.slug}/${lesson.slug}`,
    },
  }
}

export default function LessonPage({ params }: PageProps) {
  const section = getSection(params.section)
  const lesson = getLesson(params.section, params.slug)
  if (!section || !lesson) notFound()
  return <LessonShell section={section} lesson={lesson} />
}
