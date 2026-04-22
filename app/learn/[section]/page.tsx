import { notFound } from 'next/navigation'
import type { Metadata } from 'next'
import SectionLanding from '@/components/section/SectionLanding'
import { getSection, roadmap } from '@/lib/roadmap'

interface PageProps {
  params: { section: string }
}

// Pre-render every section landing at build time.
export function generateStaticParams() {
  return roadmap.map((s) => ({ section: s.slug }))
}

export function generateMetadata({ params }: PageProps): Metadata {
  const section = getSection(params.section)
  if (!section) return { title: 'Section not found' }
  return {
    title: `${section.title}`,
    description: section.subtitle,
    openGraph: {
      title: `${section.title} · ML from Scratch`,
      description: section.subtitle,
      type: 'website',
      url: `/learn/${section.slug}`,
    },
    twitter: {
      card: 'summary_large_image',
      title: `${section.title} · ML from Scratch`,
      description: section.subtitle,
    },
    alternates: {
      canonical: `/learn/${section.slug}`,
    },
  }
}

export default function SectionPage({ params }: PageProps) {
  const section = getSection(params.section)
  if (!section) notFound()
  return <SectionLanding section={section} />
}
