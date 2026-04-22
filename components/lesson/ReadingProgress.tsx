'use client'

import { useEffect, useState } from 'react'

// Thin accent bar fixed to the viewport top, filling left-to-right as the user
// reads down the lesson. Driven by a single requestAnimationFrame loop — we
// update internal state in an rAF tick rather than reacting to every scroll
// event, which can fire 60+ times a second.
export default function ReadingProgress() {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    let frame = 0
    let running = true

    function tick() {
      const doc = document.documentElement
      const scrollTop = doc.scrollTop || document.body.scrollTop
      const scrollHeight = doc.scrollHeight - doc.clientHeight
      const ratio = scrollHeight > 0 ? scrollTop / scrollHeight : 0
      setProgress(Math.min(1, Math.max(0, ratio)))
      if (running) frame = requestAnimationFrame(tick)
    }

    frame = requestAnimationFrame(tick)
    return () => {
      running = false
      cancelAnimationFrame(frame)
    }
  }, [])

  return (
    <div
      aria-hidden="true"
      className="fixed top-0 left-0 right-0 h-[2px] z-40 pointer-events-none"
    >
      <div
        className="h-full bg-dark-accent transition-[width] duration-75"
        style={{ width: `${(progress * 100).toFixed(2)}%` }}
      />
    </div>
  )
}
