'use client'

import { useEffect, useState } from 'react'
import { Search } from 'lucide-react'

// Small floating chip in the bottom-right corner that hints at the Cmd+K
// palette. Detects macOS to show ⌘K vs. Ctrl K. Hidden on narrow viewports
// where the palette is still keyboard-addressable but a chip would just clutter.
export default function CommandPaletteHint() {
  const [isMac, setIsMac] = useState(false)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    // Defer render until after hydration so the platform text is stable.
    setVisible(true)
    setIsMac(
      typeof navigator !== 'undefined' &&
        /Mac|iPhone|iPod|iPad/.test(navigator.platform),
    )
  }, [])

  if (!visible) return null

  function open() {
    window.dispatchEvent(
      new KeyboardEvent('keydown', {
        key: 'k',
        metaKey: true,
        ctrlKey: true,
        bubbles: true,
      }),
    )
  }

  return (
    <button
      type="button"
      onClick={open}
      aria-label="Open search"
      className="hidden md:flex fixed bottom-4 right-4 z-30 items-center gap-2 px-2.5 py-1.5 rounded-md border border-dark-border bg-dark-surface/90 backdrop-blur text-dark-text-disabled hover:text-dark-text-primary hover:border-dark-border-hover transition-colors font-mono text-[11px] shadow-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-dark-accent"
    >
      <Search className="w-3 h-3" />
      <span>search</span>
      <kbd className="border border-dark-border rounded px-1 py-px text-[9.5px] text-dark-text-disabled">
        {isMac ? '⌘K' : 'Ctrl K'}
      </kbd>
    </button>
  )
}
