'use client'

import { useRouter } from 'next/navigation'
import { useEffect } from 'react'
import { lessonHref } from '@/lib/utils'
import type { Lesson } from '@/lib/roadmap'

interface Props {
  prev: Lesson | null
  next: Lesson | null
}

// Global keyboard shortcut: left/right arrows jump between adjacent lessons.
// Ignored when the user is typing in an input, textarea, or contenteditable
// region, or when a modifier key is held (Cmd+Left on mac = browser back).
export default function LessonKeyboardNav({ prev, next }: Props) {
  const router = useRouter()

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return
      const target = e.target as HTMLElement | null
      if (target) {
        const tag = target.tagName
        if (
          tag === 'INPUT' ||
          tag === 'TEXTAREA' ||
          tag === 'SELECT' ||
          target.isContentEditable
        ) {
          return
        }
      }
      if (e.key === 'ArrowLeft' && prev) {
        e.preventDefault()
        router.push(lessonHref(prev.sectionSlug, prev.slug))
      } else if (e.key === 'ArrowRight' && next) {
        e.preventDefault()
        router.push(lessonHref(next.sectionSlug, next.slug))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [prev, next, router])

  return null
}
