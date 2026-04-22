'use client'

import { useEffect, useState } from 'react'
import { Check, Copy } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Props {
  /** Raw code text to copy. */
  code: string
  /** Label for screen readers. */
  label?: string
}

// Small copy-to-clipboard button, floating in the top-right of a code block.
// Shows a checkmark for 1.5s after a successful copy, falls back silently if
// the clipboard API isn't available (very old browsers, insecure context).
export default function CopyButton({ code, label = 'Copy code' }: Props) {
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) return
    const t = window.setTimeout(() => setCopied(false), 1500)
    return () => window.clearTimeout(t)
  }, [copied])

  async function onCopy() {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(code)
      } else {
        // Fallback: synthesize a hidden textarea + document.execCommand.
        const ta = document.createElement('textarea')
        ta.value = code
        ta.style.position = 'fixed'
        ta.style.top = '0'
        ta.style.left = '0'
        ta.style.opacity = '0'
        document.body.appendChild(ta)
        ta.focus()
        ta.select()
        try {
          document.execCommand('copy')
        } finally {
          document.body.removeChild(ta)
        }
      }
      setCopied(true)
    } catch {
      // Silent failure — we prefer no visible error over a noisy toast here.
    }
  }

  return (
    <button
      type="button"
      onClick={onCopy}
      aria-label={copied ? 'Copied to clipboard' : label}
      className={cn(
        'absolute top-2 right-2 z-10',
        'inline-flex items-center gap-1 px-1.5 py-1 rounded',
        'border border-dark-border bg-dark-surface/80 backdrop-blur-sm',
        'text-[10px] font-mono uppercase tracking-wider',
        'opacity-0 group-hover:opacity-100 focus-visible:opacity-100',
        'text-dark-text-disabled hover:text-dark-text-primary hover:border-dark-border-hover',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-dark-accent',
        'transition-all',
      )}
    >
      {copied ? (
        <>
          <Check className="w-3 h-3 text-term-green" />
          <span className="text-term-green">copied</span>
        </>
      ) : (
        <>
          <Copy className="w-3 h-3" />
          <span>copy</span>
        </>
      )}
    </button>
  )
}
