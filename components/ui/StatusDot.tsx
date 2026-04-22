'use client'

import { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'
import { cycleStatus, getStatus, type Status } from '@/lib/progress'

interface StatusDotProps {
  sectionSlug: string
  lessonSlug: string
  className?: string
  interactive?: boolean
}

const STYLES: Record<Status, string> = {
  'not-started': 'border-dark-border-hover bg-transparent',
  'in-progress': 'border-term-amber bg-term-amber/40',
  completed: 'border-term-green bg-term-green',
}

const LABELS: Record<Status, string> = {
  'not-started': 'Not started',
  'in-progress': 'In progress',
  completed: 'Completed',
}

// Small ring indicator that shows lesson completion state.
// When `interactive`, clicking cycles through the three states (saved to localStorage).
export default function StatusDot({
  sectionSlug,
  lessonSlug,
  className,
  interactive = false,
}: StatusDotProps) {
  const [status, setStatusLocal] = useState<Status>('not-started')

  useEffect(() => {
    setStatusLocal(getStatus(sectionSlug, lessonSlug))
    const handler = () => setStatusLocal(getStatus(sectionSlug, lessonSlug))
    window.addEventListener('mlfs:progress-updated', handler)
    return () => window.removeEventListener('mlfs:progress-updated', handler)
  }, [sectionSlug, lessonSlug])

  const dot = (
    <span
      aria-hidden
      className={cn(
        'inline-block w-2 h-2 rounded-full border transition-colors',
        STYLES[status],
        className
      )}
    />
  )

  if (!interactive) {
    return (
      <span title={LABELS[status]} className="inline-flex items-center">
        {dot}
      </span>
    )
  }

  return (
    <button
      type="button"
      aria-label={`Status: ${LABELS[status]} — click to change`}
      title={`${LABELS[status]} (click to cycle)`}
      onClick={(e) => {
        e.preventDefault()
        e.stopPropagation()
        setStatusLocal(cycleStatus(sectionSlug, lessonSlug))
      }}
      className="inline-flex items-center justify-center p-0.5 rounded hover:bg-white/[0.04] transition-colors"
    >
      {dot}
    </button>
  )
}
