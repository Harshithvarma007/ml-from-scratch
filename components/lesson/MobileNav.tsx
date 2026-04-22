'use client'

import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { Menu, X } from 'lucide-react'
import LessonSidebar from './LessonSidebar'

interface Props {
  activeSectionSlug: string
  activeLessonSlug: string
}

// Hamburger trigger visible on mobile/tablet only (md- breakpoint). Opens a
// left-side drawer containing the full LessonSidebar, rendered in a portal so
// nothing on the page can constrain it. Dismiss via: backdrop click, X button,
// Escape key, or tapping any lesson link (route change closes automatically
// because the drawer state resets when the route changes).
export default function MobileNav({ activeSectionSlug, activeLessonSlug }: Props) {
  const [open, setOpen] = useState(false)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  // Close on route change — the keys into LessonSidebar change with the route.
  useEffect(() => {
    setOpen(false)
  }, [activeSectionSlug, activeLessonSlug])

  // Lock body scroll while open.
  useEffect(() => {
    if (!open) return
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => {
      document.body.style.overflow = prev
      window.removeEventListener('keydown', onKey)
    }
  }, [open])

  const drawer =
    mounted && open
      ? createPortal(
          <div
            className="md:hidden fixed inset-0 z-50"
            role="dialog"
            aria-modal="true"
            aria-label="Curriculum navigation"
          >
            {/* Backdrop */}
            <button
              type="button"
              aria-label="Close navigation"
              onClick={() => setOpen(false)}
              className="absolute inset-0 bg-black/70 backdrop-blur-[2px]"
            />
            {/* Panel */}
            <div className="absolute inset-y-0 left-0 w-[280px] max-w-[85%] sidebar-drawer bg-dark-bg border-r border-dark-border shadow-2xl flex flex-col">
              <div className="flex items-center justify-end px-2 py-2 border-b border-dark-border">
                <button
                  type="button"
                  onClick={() => setOpen(false)}
                  aria-label="Close navigation"
                  className="p-1.5 rounded hover:bg-white/[0.05] transition-colors text-dark-text-disabled hover:text-dark-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-dark-accent"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="flex-1 min-h-0">
                <LessonSidebar
                  activeSectionSlug={activeSectionSlug}
                  activeLessonSlug={activeLessonSlug}
                />
              </div>
            </div>
          </div>,
          document.body,
        )
      : null

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label="Open curriculum navigation"
        className="md:hidden fixed top-3 left-3 z-40 p-2 rounded-md bg-dark-surface border border-dark-border text-dark-text-secondary hover:text-dark-text-primary hover:border-dark-border-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-dark-accent transition-colors"
      >
        <Menu className="w-4 h-4" />
      </button>
      {drawer}
    </>
  )
}
