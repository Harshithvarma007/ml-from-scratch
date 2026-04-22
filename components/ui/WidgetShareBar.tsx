'use client'

import { useState } from 'react'
import { Link2, Code2, ArrowLeft } from 'lucide-react'
import Link from 'next/link'

const SITE_URL =
  typeof window !== 'undefined'
    ? window.location.origin
    : (process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.harshithvarma.in')

function useCopy(text: string) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return { copy, copied }
}

export default function WidgetShareBar({ widgetName }: { widgetName: string }) {
  const widgetUrl = `${SITE_URL}/widget/${widgetName}`
  const embedSnippet = `<iframe src="${widgetUrl}" width="800" height="500" style="border:none;border-radius:8px;" title="${widgetName} — ML from Scratch" loading="lazy"></iframe>`

  const { copy: copyLink, copied: linkCopied } = useCopy(widgetUrl)
  const { copy: copyEmbed, copied: embedCopied } = useCopy(embedSnippet)

  return (
    <header className="border-b border-dark-border bg-dark-surface/80 backdrop-blur px-6 py-2 flex items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <Link
          href="/"
          className="flex items-center gap-1.5 font-mono text-[11px] text-dark-text-muted hover:text-dark-text-primary transition-colors"
        >
          <ArrowLeft className="w-3 h-3" />
          ml·from·scratch
        </Link>
        <span className="text-dark-border">·</span>
        <span className="font-mono text-[11px] text-dark-text-secondary">{widgetName}</span>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={copyLink}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded font-mono text-[11px] border border-dark-border text-dark-text-secondary hover:text-dark-text-primary hover:border-dark-border-hover transition-all"
          title="Copy shareable link"
        >
          <Link2 className="w-3 h-3" />
          {linkCopied ? 'copied!' : 'copy link'}
        </button>
        <button
          onClick={copyEmbed}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded font-mono text-[11px] border border-dark-border text-dark-text-secondary hover:text-dark-text-primary hover:border-dark-border-hover transition-all"
          title="Copy iframe embed snippet"
        >
          <Code2 className="w-3 h-3" />
          {embedCopied ? 'copied!' : 'embed'}
        </button>
      </div>
    </header>
  )
}
