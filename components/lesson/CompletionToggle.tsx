'use client'

import { useEffect, useState } from 'react'
import { Check, Circle, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { cycleStatus, getStatus, setStatus as setProgress, type Status } from '@/lib/progress'

interface CompletionToggleProps {
  sectionSlug: string
  lessonSlug: string
}

const CONFIG: Record<Status, { label: string; icon: typeof Check; classes: string }> = {
  'not-started': {
    label: 'Mark in progress',
    icon: Circle,
    classes: 'border-dark-border text-dark-text-muted hover:text-dark-text-primary',
  },
  'in-progress': {
    label: 'Mark completed',
    icon: Loader2,
    classes: 'border-term-amber/40 text-term-amber hover:bg-term-amber/5',
  },
  completed: {
    label: 'Completed · reset',
    icon: Check,
    classes: 'border-term-green/40 text-term-green hover:bg-term-green/5',
  },
}

// Inline progress control shown right below the lesson header.
// Reads/writes localStorage via lib/progress.
export function CompletionToggle({ sectionSlug, lessonSlug }: CompletionToggleProps) {
  const [status, setStatusLocal] = useState<Status>('not-started')
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    setStatusLocal(getStatus(sectionSlug, lessonSlug))
    const handler = () => setStatusLocal(getStatus(sectionSlug, lessonSlug))
    window.addEventListener('mlfs:progress-updated', handler)
    return () => window.removeEventListener('mlfs:progress-updated', handler)
  }, [sectionSlug, lessonSlug])

  if (!mounted) {
    // Avoid hydration mismatch — render a neutral placeholder on the server.
    return <div className="h-8 mb-6" aria-hidden />
  }

  const cfg = CONFIG[status]
  const Icon = cfg.icon

  return (
    <div className="mb-6 flex items-center gap-3">
      <button
        type="button"
        onClick={() => setStatusLocal(cycleStatus(sectionSlug, lessonSlug))}
        className={cn(
          'inline-flex items-center gap-2 px-3 py-1.5 rounded-md border',
          'font-mono text-[11px] uppercase tracking-wider transition-colors',
          cfg.classes
        )}
      >
        <Icon
          className={cn('w-3.5 h-3.5', status === 'in-progress' && 'animate-spin-slow')}
          strokeWidth={1.75}
        />
        {cfg.label}
      </button>
      {status !== 'not-started' && (
        <button
          type="button"
          onClick={() => {
            setProgress(sectionSlug, lessonSlug, 'not-started')
            setStatusLocal('not-started')
          }}
          className="text-[10px] font-mono text-dark-text-disabled hover:text-dark-text-secondary transition-colors uppercase tracking-wider"
        >
          reset
        </button>
      )}
    </div>
  )
}
