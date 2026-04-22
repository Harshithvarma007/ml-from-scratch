'use client'

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useRouter } from 'next/navigation'
import { Search, ArrowRight, CornerDownLeft, Sparkles } from 'lucide-react'
import { accentClasses, flatLessons, roadmap } from '@/lib/roadmap'
import { cn, lessonHref } from '@/lib/utils'

interface Entry {
  kind: 'lesson' | 'section'
  sectionSlug: string
  sectionTitle: string
  sectionAccent: keyof typeof accentClasses
  lessonSlug?: string
  title: string
  blurb: string
  href: string
  // Extra search terms beyond title/blurb — synonyms, acronyms. Never shown.
  keywords: string[]
}

// Marquee lessons surfaced in the empty-state "start here" list. Hand-picked
// so a first-time opener of Cmd+K sees the canonical entry points rather than
// an exhaustive ranked dump.
const FEATURED_SLUGS = [
  'gradient-descent',
  'backpropagation',
  'self-attention',
  'training-loop',
] as const

// Build the search index once per render of the root. It's a flat list of
// every section and lesson, pre-formatted for fuzzy matching.
function buildIndex(): Entry[] {
  const entries: Entry[] = []
  for (const s of roadmap) {
    entries.push({
      kind: 'section',
      sectionSlug: s.slug,
      sectionTitle: s.title,
      sectionAccent: s.accent,
      title: s.title,
      blurb: s.subtitle,
      href: `/learn/${s.slug}`,
      keywords: [],
    })
  }
  for (const l of flatLessons()) {
    const section = roadmap.find((s) => s.slug === l.sectionSlug)!
    entries.push({
      kind: 'lesson',
      sectionSlug: l.sectionSlug,
      sectionTitle: section.title,
      sectionAccent: section.accent,
      lessonSlug: l.slug,
      title: l.title,
      blurb: l.blurb,
      href: lessonHref(l.sectionSlug, l.slug),
      keywords: l.keywords ?? [],
    })
  }
  return entries
}

// Match query against an entry using a light fuzzy-scoring heuristic:
// +100 prefix match on title, +50 substring on title, +20 substring on blurb,
// +15 substring on a keyword, +10 substring on section title.
// Returns 0 when there's no match (empty query is handled by the caller, which
// shows a curated "start here" list instead of scoring everything at 1).
function scoreEntry(query: string, e: Entry): number {
  if (!query) return 0
  const q = query.toLowerCase()
  const t = e.title.toLowerCase()
  const b = e.blurb.toLowerCase()
  const s = e.sectionTitle.toLowerCase()
  let score = 0
  if (t.startsWith(q)) score += 100
  else if (t.includes(q)) score += 50
  if (b.includes(q)) score += 20
  if (e.keywords.some((k) => k.toLowerCase().includes(q))) score += 15
  if (s.includes(q)) score += 10
  // Token fallback: every whitespace-separated token must be present somewhere
  // (including keywords), so queries like "chain rule" can hit Backpropagation.
  const tokens = q.split(/\s+/).filter(Boolean)
  if (tokens.length > 1) {
    const all = `${t} ${b} ${s} ${e.keywords.join(' ').toLowerCase()}`
    if (tokens.every((tok) => all.includes(tok))) score = Math.max(score, 25)
  }
  return score
}

// Wrap every case-insensitive occurrence of `query` inside `text` with a
// <mark>, returning a React fragment. Empty query returns text unchanged.
// Kept small and dependency-free — no regex escaping gymnastics for edge
// punctuation since search queries are plain words.
function highlight(text: string, query: string) {
  if (!query) return text
  const q = query.trim()
  if (!q) return text
  const lower = text.toLowerCase()
  const needle = q.toLowerCase()
  const parts: React.ReactNode[] = []
  let cursor = 0
  let idx = lower.indexOf(needle, cursor)
  while (idx !== -1) {
    if (idx > cursor) parts.push(text.slice(cursor, idx))
    parts.push(
      <mark
        key={idx}
        className="bg-dark-accent/25 text-dark-accent rounded px-0.5"
      >
        {text.slice(idx, idx + needle.length)}
      </mark>,
    )
    cursor = idx + needle.length
    idx = lower.indexOf(needle, cursor)
  }
  if (cursor < text.length) parts.push(text.slice(cursor))
  return <Fragment>{parts}</Fragment>
}

export default function CommandPalette() {
  const router = useRouter()
  const [open, setOpen] = useState(false)
  const [mounted, setMounted] = useState(false)
  const [query, setQuery] = useState('')
  const [activeIdx, setActiveIdx] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setMounted(true)
  }, [])

  const index = useMemo(buildIndex, [])

  // When the query is empty, the palette shows a curated landing view:
  // a handful of marquee lessons ("start here") followed by every section.
  // When there IS a query, we score and sort everything instead. Keeping the
  // two code paths separate makes the empty state feel deliberate rather than
  // like a noisy default ranking.
  const { results, isFeatured } = useMemo(() => {
    if (!query.trim()) {
      const featured: Entry[] = []
      for (const slug of FEATURED_SLUGS) {
        const hit = index.find(
          (e) => e.kind === 'lesson' && e.lessonSlug === slug,
        )
        if (hit) featured.push(hit)
      }
      const sections = index.filter((e) => e.kind === 'section')
      return { results: [...featured, ...sections], isFeatured: true }
    }
    const scored = index
      .map((e) => ({ e, score: scoreEntry(query, e) }))
      .filter((r) => r.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 40)
    return { results: scored.map((r) => r.e), isFeatured: false }
  }, [index, query])

  // Index at which the "All sections" group begins in the featured view — we
  // inject a divider label there. Computed from FEATURED_SLUGS so it stays in
  // sync if the featured list changes.
  const featuredCount = isFeatured ? FEATURED_SLUGS.length : 0

  // Global Cmd/Ctrl+K listener
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.key === 'k' || e.key === 'K') && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        setOpen((o) => !o)
        return
      }
      if (e.key === '/' && !open) {
        const target = e.target as HTMLElement | null
        const inForm =
          target &&
          (target.tagName === 'INPUT' ||
            target.tagName === 'TEXTAREA' ||
            target.isContentEditable)
        if (!inForm) {
          e.preventDefault()
          setOpen(true)
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  // Reset search + focus input when opening
  useEffect(() => {
    if (open) {
      setQuery('')
      setActiveIdx(0)
      requestAnimationFrame(() => inputRef.current?.focus())
    }
  }, [open])

  // Clamp active index when results shrink
  useEffect(() => {
    if (activeIdx >= results.length) setActiveIdx(0)
  }, [results, activeIdx])

  // Keep active item in view
  useEffect(() => {
    if (!open) return
    const list = listRef.current
    if (!list) return
    const active = list.querySelector<HTMLButtonElement>(
      `[data-cmd-idx="${activeIdx}"]`,
    )
    if (active) active.scrollIntoView({ block: 'nearest' })
  }, [activeIdx, open])

  const select = useCallback(
    (entry: Entry) => {
      setOpen(false)
      router.push(entry.href)
    },
    [router],
  )

  const onInputKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        setOpen(false)
        return
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setActiveIdx((i) => Math.min(results.length - 1, i + 1))
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setActiveIdx((i) => Math.max(0, i - 1))
      } else if (e.key === 'Enter') {
        e.preventDefault()
        const entry = results[activeIdx]
        if (entry) select(entry)
      }
    },
    [results, activeIdx, select],
  )

  if (!mounted || !open) return null

  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Search curriculum"
      className="fixed inset-0 z-[60] flex items-start justify-center pt-[10vh] px-4"
    >
      {/* Backdrop */}
      <button
        type="button"
        aria-label="Close search"
        onClick={() => setOpen(false)}
        className="absolute inset-0 bg-black/70 backdrop-blur-[2px]"
      />

      {/* Panel */}
      <div className="relative w-full max-w-xl rounded-lg border border-dark-border bg-dark-surface shadow-2xl overflow-hidden animate-fadeIn">
        {/* Input */}
        <div className="flex items-center gap-2.5 px-3.5 py-2.5 border-b border-dark-border">
          <Search className="w-4 h-4 text-dark-text-disabled" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onInputKeyDown}
            placeholder="Search lessons, sections…"
            className="flex-1 bg-transparent outline-none border-none font-mono text-[13px] text-dark-text-primary placeholder:text-dark-text-disabled"
            aria-label="Search curriculum"
            aria-controls="command-palette-results"
            aria-activedescendant={
              results[activeIdx] ? `cmd-item-${activeIdx}` : undefined
            }
          />
          <kbd className="font-mono text-[10px] uppercase tracking-wider text-dark-text-disabled border border-dark-border rounded px-1.5 py-0.5">
            esc
          </kbd>
        </div>

        {/* Results */}
        <div
          ref={listRef}
          id="command-palette-results"
          role="listbox"
          className="max-h-[55vh] overflow-y-auto"
        >
          {results.length === 0 ? (
            <div className="px-3.5 py-8 text-center font-mono text-[12px] text-dark-text-disabled">
              no matches for &ldquo;{query}&rdquo;. try a shorter phrase.
            </div>
          ) : (
            <>
              {isFeatured && (
                <div className="flex items-center gap-1.5 px-3.5 pt-2.5 pb-1 font-mono text-[9.5px] uppercase tracking-wider text-dark-text-disabled">
                  <Sparkles className="w-2.5 h-2.5" strokeWidth={2} />
                  start here
                </div>
              )}
              {results.map((e, i) => {
                const accent = accentClasses[e.sectionAccent]
                const active = i === activeIdx
                const showSectionsDivider = isFeatured && i === featuredCount
                return (
                  <Fragment key={`${e.kind}-${e.sectionSlug}-${e.lessonSlug ?? ''}`}>
                    {showSectionsDivider && (
                      <div className="px-3.5 pt-3 pb-1 font-mono text-[9.5px] uppercase tracking-wider text-dark-text-disabled border-t border-dark-border/50 mt-1">
                        all sections
                      </div>
                    )}
                    <button
                      id={`cmd-item-${i}`}
                      data-cmd-idx={i}
                      role="option"
                      aria-selected={active}
                      onClick={() => select(e)}
                      onMouseEnter={() => setActiveIdx(i)}
                      className={cn(
                        'w-full flex items-center gap-3 px-3.5 py-2 text-left border-l-2 transition-colors',
                        active
                          ? 'bg-white/[0.03] border-l-dark-accent'
                          : 'border-l-transparent hover:bg-white/[0.02]',
                      )}
                    >
                      <span
                        className={cn(
                          'w-1.5 h-1.5 rounded-full flex-shrink-0',
                          accent.dot,
                        )}
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span
                            className={cn(
                              'font-mono text-[12.5px] truncate',
                              active
                                ? 'text-dark-text-primary'
                                : 'text-dark-text-secondary',
                            )}
                          >
                            {highlight(e.title, query)}
                          </span>
                          {e.kind === 'section' && (
                            <span className="text-[9.5px] font-mono uppercase tracking-wider text-dark-text-disabled border border-dark-border rounded px-1 py-px">
                              section
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-2 mt-0.5">
                          <span
                            className={cn(
                              'text-[10px] font-mono uppercase tracking-wider',
                              accent.text,
                            )}
                          >
                            {highlight(e.sectionTitle, query)}
                          </span>
                          <span className="text-[10px] text-dark-text-disabled truncate">
                            {highlight(e.blurb, query)}
                          </span>
                        </div>
                      </div>
                      {active && (
                        <ArrowRight className="w-3.5 h-3.5 text-dark-accent flex-shrink-0" />
                      )}
                    </button>
                  </Fragment>
                )
              })}
            </>
          )}
        </div>

        {/* Footer hint */}
        <div className="flex items-center justify-between gap-3 px-3.5 py-2 border-t border-dark-border bg-dark-surface-elevated/40 font-mono text-[10px] text-dark-text-disabled">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <kbd className="border border-dark-border rounded px-1 py-px">↑</kbd>
              <kbd className="border border-dark-border rounded px-1 py-px">↓</kbd>
              move
            </span>
            <span className="flex items-center gap-1">
              <kbd className="border border-dark-border rounded px-1 py-px">
                <CornerDownLeft className="inline w-2.5 h-2.5" />
              </kbd>
              open
            </span>
          </div>
          <span>
            {isFeatured
              ? 'type to search 75 lessons'
              : `${results.length} match${results.length === 1 ? '' : 'es'}`}
          </span>
        </div>
      </div>
    </div>,
    document.body,
  )
}
