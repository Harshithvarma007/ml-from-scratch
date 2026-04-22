import { ImageResponse } from 'next/og'
import { flatLessons, getLesson, getSection } from '@/lib/roadmap'

export const runtime = 'edge'
export const alt = 'ML from Scratch — Lesson'
export const size = { width: 1200, height: 630 }
export const contentType = 'image/png'

export function generateImageMetadata({
  params,
}: {
  params: { section: string; slug: string }
}) {
  const lesson = getLesson(params.section, params.slug)
  return [
    {
      id: `${params.section}-${params.slug}`,
      alt: lesson ? `${lesson.title} — ML from Scratch` : alt,
      contentType,
      size,
    },
  ]
}

export function generateStaticParams() {
  return flatLessons().map((l) => ({ section: l.sectionSlug, slug: l.slug }))
}

const DIFF_COLOR: Record<string, string> = {
  Easy: '#4ade80',
  Medium: '#fbbf24',
  Hard: '#f87171',
}

const DIFF_BG: Record<string, string> = {
  Easy: 'rgba(74,222,128,0.08)',
  Medium: 'rgba(251,191,36,0.08)',
  Hard: 'rgba(248,113,113,0.08)',
}

export default function LessonOGImage({
  params,
}: {
  params: { section: string; slug: string }
}) {
  const section = getSection(params.section)
  const lesson = getLesson(params.section, params.slug)

  if (!section || !lesson) {
    return new ImageResponse(
      (
        <div style={{
          width: '100%', height: '100%', display: 'flex',
          alignItems: 'center', justifyContent: 'center',
          backgroundColor: '#080808', color: '#e8e8e8',
          fontSize: 64, fontFamily: 'monospace',
        }}>
          ML from Scratch
        </div>
      ),
      { ...size },
    )
  }

  const diffColor = DIFF_COLOR[lesson.difficulty] ?? '#888'
  const diffBg = DIFF_BG[lesson.difficulty] ?? 'transparent'

  return new ImageResponse(
    (
      <div style={{
        width: '100%', height: '100%', display: 'flex',
        flexDirection: 'column', justifyContent: 'space-between',
        backgroundColor: '#080808', fontFamily: 'monospace',
        position: 'relative',
      }}>
        {/* Top accent line */}
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, height: 3,
          background: 'linear-gradient(90deg, transparent, #a78bfa 30%, #a78bfa 70%, transparent)',
        }} />

        {/* Grid dots */}
        <div style={{
          position: 'absolute', inset: 0,
          backgroundImage: 'radial-gradient(circle, #1a1a1a 1px, transparent 1px)',
          backgroundSize: '40px 40px',
          opacity: 0.5,
        }} />

        {/* Left accent bar */}
        <div style={{
          position: 'absolute', left: 0, top: 0, bottom: 0, width: 4,
          background: `linear-gradient(180deg, transparent, ${diffColor}60, transparent)`,
        }} />

        {/* Content */}
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height: '100%', padding: '64px 72px', position: 'relative' }}>

          {/* Top: breadcrumb */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ color: '#a78bfa', fontSize: 20 }}>ml ›</span>
            <span style={{ color: '#444', fontSize: 20 }}>{section.title}</span>
            <span style={{ color: '#2a2a2a', fontSize: 20 }}>›</span>
            <span style={{ color: '#555', fontSize: 20 }}>{lesson.title}</span>
          </div>

          {/* Middle: lesson title + blurb */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div style={{
              color: '#e8e8e8', fontSize: lesson.title.length > 30 ? 68 : 80,
              fontWeight: 700, lineHeight: 1.05, letterSpacing: -1,
            }}>
              {lesson.title}
            </div>
            <div style={{ color: '#666', fontSize: 26, lineHeight: 1.4, maxWidth: 800 }}>
              {lesson.blurb}
            </div>
          </div>

          {/* Bottom: difficulty + attribution */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '10px 20px',
              border: `1px solid ${diffColor}40`,
              borderRadius: 6,
              backgroundColor: diffBg,
              color: diffColor,
              fontSize: 20,
            }}>
              <span style={{ opacity: 0.6 }}>difficulty</span>
              <span style={{ color: '#333' }}>·</span>
              <span style={{ fontWeight: 600 }}>{lesson.difficulty.toUpperCase()}</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
              <span style={{ color: '#a78bfa', fontSize: 18, fontWeight: 600 }}>ML from Scratch</span>
              <span style={{ color: '#333', fontSize: 16 }}>ml.harshithvarma.in</span>
            </div>
          </div>
        </div>
      </div>
    ),
    { ...size },
  )
}
