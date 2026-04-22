import {
  AlertTriangle,
  ArrowRight,
  BookOpen,
  Code2,
  ExternalLink,
  FileText,
  Link2,
  Sigma,
  Terminal,
  Zap,
} from 'lucide-react'
import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

// ---------- Prose ----------
// Long-form reading text. Switches to Inter sans for comfort — lesson bodies
// rotate between mono (math, code, terminal feel) and sans (explanation).
export function Prose({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        'font-sans text-[14px] leading-[1.7] text-dark-text-primary',
        'space-y-4',
        '[&_strong]:font-semibold [&_strong]:text-dark-text-primary',
        '[&_em]:text-dark-text-secondary [&_em]:italic',
        '[&_code]:font-mono [&_code]:text-[13px] [&_code]:px-1 [&_code]:py-0.5',
        '[&_code]:bg-dark-surface-elevated [&_code]:border [&_code]:border-dark-border',
        '[&_code]:rounded [&_code]:text-dark-text-primary',
        '[&_a]:text-dark-accent [&_a]:underline [&_a]:underline-offset-2 [&_a]:decoration-dark-accent/30',
        '[&_a:hover]:decoration-dark-accent',
        '[&_ul]:list-none [&_ul]:space-y-1.5 [&_ul]:pl-0',
        '[&_ul>li]:pl-4 [&_ul>li]:relative',
        '[&_ul>li]:before:content-["›"] [&_ul>li]:before:absolute [&_ul>li]:before:left-0',
        '[&_ul>li]:before:text-dark-accent [&_ul>li]:before:font-mono',
        className
      )}
    >
      {children}
    </div>
  )
}

// ---------- Callout ----------
// Side-note / observation block. Small, left-rule accented, non-obtrusive.
export function Callout({
  children,
  variant = 'note',
  title,
}: {
  children: ReactNode
  variant?: 'note' | 'insight' | 'warn'
  title?: string
}) {
  const accent = {
    note: 'border-l-dark-accent/50 bg-dark-accent/[0.04]',
    insight: 'border-l-term-cyan/50 bg-term-cyan/[0.04]',
    warn: 'border-l-term-amber/60 bg-term-amber/[0.05]',
  }[variant]
  const label = {
    note: 'note',
    insight: 'insight',
    warn: 'heads up',
  }[variant]
  const iconClass = {
    note: 'text-dark-accent',
    insight: 'text-term-cyan',
    warn: 'text-term-amber',
  }[variant]

  return (
    <aside className={cn('border-l-2 rounded-r px-4 py-3 my-5', accent)}>
      <div className="flex items-center gap-1.5 mb-1.5">
        <Zap className={cn('w-3 h-3', iconClass)} strokeWidth={2} />
        <span className={cn('text-[10px] uppercase tracking-wider font-mono', iconClass)}>
          {title ?? label}
        </span>
      </div>
      <div className="font-sans text-[13px] leading-relaxed text-dark-text-secondary">
        {children}
      </div>
    </aside>
  )
}

// ---------- Personify ----------
// One-liner in the voice of a personified concept ("Gradient: I point uphill.").
// A signature move of the house style — small, italic, colored.
export function Personify({ speaker, children }: { speaker: string; children: ReactNode }) {
  return (
    <figure className="my-5 border-l-2 border-term-purple/60 bg-term-purple/[0.05] pl-4 py-2 pr-3">
      <figcaption className="text-[10px] uppercase tracking-wider font-mono text-term-purple mb-0.5">
        {speaker} (personified)
      </figcaption>
      <blockquote className="font-sans text-[13.5px] italic text-dark-text-primary leading-relaxed">
        {children}
      </blockquote>
    </figure>
  )
}

// ---------- Bridge ----------
// Explicit ←→ mapping between two representations. Each row = one bridge line.
export interface BridgeRow {
  left: string
  right: string
  note?: string
}

export function Bridge({ rows, label }: { rows: BridgeRow[]; label?: string }) {
  return (
    <div className="term-panel rounded-lg overflow-hidden my-5">
      <header className="term-panel-header">
        <div className="flex items-center gap-2">
          <ArrowRight className="w-3 h-3 text-dark-accent" strokeWidth={2} />
          <span>{label ?? 'bridge'}</span>
        </div>
      </header>
      <div className="p-4 font-mono text-[12.5px] space-y-2">
        {rows.map((row, i) => (
          <div key={i} className="grid grid-cols-[1fr_auto_1fr] gap-3 items-start">
            <code className="text-dark-text-primary">{row.left}</code>
            <span className="text-dark-accent select-none pt-0.5">←→</span>
            <code className="text-dark-text-primary">{row.right}</code>
            {row.note && (
              <p className="col-span-3 text-[11px] text-dark-text-muted pl-2 -mt-1">
                <span className="text-dark-text-disabled">— </span>
                {row.note}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

// ---------- Gotcha ----------
// Bulleted edge-case list — the kind with scary-but-specific examples.
export function Gotcha({ children }: { children: ReactNode }) {
  return (
    <div className="term-panel rounded-lg overflow-hidden my-5 border-term-amber/30">
      <header className="term-panel-header text-term-amber">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-3 h-3" strokeWidth={2} />
          <span>Gotchas</span>
        </div>
      </header>
      <div className="p-4 font-sans text-[13px] text-dark-text-secondary leading-relaxed space-y-2.5">
        {children}
      </div>
    </div>
  )
}

// ---------- Challenge ----------
// End-of-lesson runnable experiment the reader can try right now.
export function Challenge({ children, prompt }: { children: ReactNode; prompt?: string }) {
  return (
    <div className="term-panel rounded-lg overflow-hidden my-5 border-term-green/30">
      <header className="term-panel-header text-term-green">
        <div className="flex items-center gap-2">
          <Terminal className="w-3 h-3" strokeWidth={2} />
          <span>{prompt ?? 'Try this'}</span>
        </div>
      </header>
      <div className="p-4 font-sans text-[13px] text-dark-text-primary leading-relaxed">
        {children}
      </div>
    </div>
  )
}

// ---------- References ----------
export type ReferenceTag = 'paper' | 'blog' | 'code' | 'book'

export interface Reference {
  title: string
  author?: string
  venue?: string
  year?: number
  url?: string
  /** One or more topical tags. Surfaces an icon + label pill next to the title. */
  tags?: ReferenceTag[]
}

const TAG_META: Record<
  ReferenceTag,
  { icon: typeof FileText; label: string; color: string }
> = {
  paper: { icon: FileText, label: 'paper', color: 'text-term-cyan' },
  blog: { icon: Link2, label: 'blog', color: 'text-term-amber' },
  code: { icon: Code2, label: 'code', color: 'text-term-green' },
  book: { icon: BookOpen, label: 'book', color: 'text-term-purple' },
}

export function References({ items }: { items: Reference[] }) {
  return (
    <div className="term-panel rounded-lg overflow-hidden my-5">
      <header className="term-panel-header">
        <div className="flex items-center gap-2">
          <BookOpen className="w-3 h-3 text-dark-accent" strokeWidth={2} />
          <span>References</span>
        </div>
      </header>
      <ul className="p-4 space-y-2.5 font-sans text-[13px] text-dark-text-secondary leading-relaxed">
        {items.map((ref, i) => (
          <li key={i} className="flex gap-3">
            <span className="text-[10px] tabular-nums text-dark-text-disabled font-mono pt-1">
              [{String(i + 1).padStart(2, '0')}]
            </span>
            <div className="flex-1 min-w-0">
              <div className="flex items-start gap-2 flex-wrap">
                {ref.url ? (
                  <a
                    href={ref.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group inline-flex items-center gap-1 text-dark-text-primary hover:text-dark-accent underline decoration-dark-border hover:decoration-dark-accent underline-offset-2 transition-colors"
                  >
                    <span>{ref.title}</span>
                    <ExternalLink
                      className="w-3 h-3 text-dark-text-disabled group-hover:text-dark-accent shrink-0"
                      aria-hidden="true"
                    />
                  </a>
                ) : (
                  <span className="text-dark-text-primary">{ref.title}</span>
                )}
                {ref.tags && ref.tags.length > 0 && (
                  <span className="flex items-center gap-1 flex-wrap">
                    {ref.tags.map((tag) => {
                      const meta = TAG_META[tag]
                      const Icon = meta.icon
                      return (
                        <span
                          key={tag}
                          className={cn(
                            'inline-flex items-center gap-0.5 px-1 py-0 rounded',
                            'border border-dark-border bg-dark-surface-elevated/40',
                            'text-[9px] font-mono uppercase tracking-wider',
                            meta.color,
                          )}
                          aria-label={`Type: ${meta.label}`}
                        >
                          <Icon className="w-2.5 h-2.5" strokeWidth={2} />
                          <span>{meta.label}</span>
                        </span>
                      )
                    })}
                  </span>
                )}
              </div>
              {(ref.author || ref.venue || ref.year) && (
                <div className="text-[12px] text-dark-text-muted mt-0.5">
                  {[ref.author, ref.venue, ref.year].filter(Boolean).join(' · ')}
                </div>
              )}
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}

// ---------- AsciiBlock ----------
// Low-fi monospace diagram — the training-loop flowchart kind.
export function AsciiBlock({ children, caption }: { children: string; caption?: string }) {
  return (
    <figure className="my-5">
      <pre
        className={cn(
          'term-panel-elevated rounded-lg p-5 overflow-x-auto',
          'font-mono text-[12px] leading-[1.65] text-dark-text-primary',
          'whitespace-pre'
        )}
      >
        {children}
      </pre>
      {caption && (
        <figcaption className="text-[11px] font-mono text-dark-text-muted mt-2 text-center">
          {caption}
        </figcaption>
      )}
    </figure>
  )
}

// ---------- KeyTerm ----------
// First-mention bolding for a defined term. Small style cue, big for scan-ability.
export function KeyTerm({ children }: { children: ReactNode }) {
  return <strong className="text-dark-accent font-semibold">{children}</strong>
}

// ---------- Eq / EqRef ----------
// Numbered equation block + cross-reference anchor. Mirrors the d2l textbook
// pattern of calling out important equations by number and letting prose refer
// back to them — "combining (5.3) with (5.5) gives us…" — without forcing the
// reader to scroll-hunt for the expression.
//
// Authors supply a semantic `id` (the anchor target) and a human-readable
// `number` (shown in parens to the right of the equation). The number is
// explicit rather than auto-computed because the lesson tree is server-
// rendered across file boundaries — a runtime counter would be fragile. In
// practice numbers are cheap to keep consistent within a single lesson.
export interface EqProps {
  children: string
  id: string
  number: string
  caption?: string
}

export function Eq({ children, id, number, caption }: EqProps) {
  return (
    <figure
      id={`eq-${id}`}
      className="term-panel-elevated rounded-lg my-5 overflow-hidden scroll-mt-24"
    >
      {caption && (
        <div className="px-4 pt-3 flex items-center gap-1.5">
          <Sigma className="w-3 h-3 text-dark-accent" strokeWidth={2} />
          <span className="text-[10px] uppercase tracking-wider font-mono text-dark-text-secondary">
            {caption}
          </span>
        </div>
      )}
      <div className="flex items-stretch">
        <pre
          className={cn(
            'flex-1 px-5 py-4 overflow-x-auto',
            'font-mono text-[13px] leading-[1.8] text-dark-text-primary whitespace-pre',
          )}
        >
          {children.replace(/\n$/, '')}
        </pre>
        <div className="shrink-0 flex items-center px-4 border-l border-dark-border/60">
          <span className="text-[11px] font-mono text-dark-text-disabled tabular-nums select-none">
            ({number})
          </span>
        </div>
      </div>
    </figure>
  )
}

// Inline reference to an Eq block. Renders "(5.3)" as an anchor link that
// scrolls smoothly to the equation above (scroll-mt handles header offset).
export function EqRef({ id, number }: { id: string; number: string }) {
  return (
    <a
      href={`#eq-${id}`}
      className={cn(
        'font-mono text-[12.5px] tabular-nums',
        'text-dark-accent underline decoration-dark-accent/30',
        'hover:decoration-dark-accent underline-offset-2 transition-colors',
      )}
      aria-label={`Equation ${number}`}
    >
      ({number})
    </a>
  )
}
