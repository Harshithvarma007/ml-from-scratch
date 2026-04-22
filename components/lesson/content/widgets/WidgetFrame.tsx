'use client'

import { useCallback, useEffect, useId, useRef, useState, type ReactNode } from 'react'
import { HelpCircle, Link2 } from 'lucide-react'
import { cn } from '@/lib/utils'

// Shared frame around every interactive widget. Term-panel aesthetic, optional
// top label, body for the canvas/SVG, footer for controls.
//
// Accessibility notes:
// - The whole widget is a `role="region"` with its label exposed as aria-label
//   so screen readers can list widgets by topic.
// - `helpText` wires in a small "?" trigger in the header that reveals a
//   popover with a lesson-side recap — keeps explanations close to the thing
//   being explained.
interface WidgetFrameProps {
  label?: string
  right?: ReactNode
  children: ReactNode
  controls?: ReactNode
  aspect?: 'square' | 'wide' | 'tall'
  className?: string
  /** Short recap of what the widget shows. Renders as a "?" popover in the header. */
  helpText?: ReactNode
  /** Forwarded aria-label. Defaults to `label` if both are strings. */
  ariaLabel?: string
  /** PascalCase widget name — when set, a share icon links to /widget/[name] */
  widgetName?: string
}

export default function WidgetFrame({
  label,
  right,
  children,
  controls,
  aspect = 'wide',
  className,
  helpText,
  ariaLabel,
  widgetName,
}: WidgetFrameProps) {
  const aspectClass = {
    square: 'aspect-square',
    wide: 'aspect-[16/10]',
    tall: 'aspect-[4/5]',
  }[aspect]

  const resolvedAria =
    ariaLabel ?? (typeof label === 'string' ? label : undefined)

  return (
    <figure
      role="region"
      aria-label={resolvedAria}
      className={cn('term-panel rounded-lg overflow-hidden my-6', className)}
    >
      {(label || right || helpText || widgetName) && (
        <header className="term-panel-header">
          <div className="flex items-center gap-2">
            {label && <span>{label}</span>}
            {helpText && <HelpPopover content={helpText} />}
          </div>
          <div className="flex items-center gap-3 text-[10px] normal-case tracking-normal text-dark-text-disabled font-mono">
            {right}
            {widgetName && <ShareLink widgetName={widgetName} />}
          </div>
        </header>
      )}
      <div className={cn('relative bg-dark-bg', aspectClass, 'w-full')}>{children}</div>
      {controls && (
        <div className="border-t border-dark-border bg-dark-surface-elevated/40 px-4 py-3">
          {controls}
        </div>
      )}
    </figure>
  )
}

// Slider styled for the terminal palette. Improvements over the previous version:
// - `aria-label` (from `label`) and `aria-valuetext` (from `format(value)`) so
//   screen readers announce a human-readable value, not the raw number.
// - Shift+Arrow bumps by 10× step, matching macOS/Windows slider conventions.
// - Slightly taller hit area (8px) for easier mouse/touch targets.
interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (n: number) => void
  format?: (n: number) => string
  accent?: string
  /** Optional longer description for the input, exposed to assistive tech. */
  description?: string
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 0.01,
  onChange,
  format,
  accent = 'accent-dark-accent',
  description,
}: SliderProps) {
  const formatted = format ? format(value) : value.toFixed(2)

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      // Native range already handles ArrowLeft/Right/Up/Down and PageUp/Down.
      // We intercept Shift+Arrow to add a 10× step, which the native behavior
      // doesn't do consistently across browsers.
      if (!e.shiftKey) return
      let delta = 0
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') delta = -step * 10
      else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') delta = step * 10
      if (delta === 0) return
      e.preventDefault()
      const next = Math.max(min, Math.min(max, value + delta))
      onChange(next)
    },
    [value, min, max, step, onChange],
  )

  return (
    <label className="flex items-center gap-3 flex-1 min-w-0 font-mono text-[12px]">
      <span className="text-dark-text-secondary whitespace-nowrap">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        onKeyDown={onKeyDown}
        aria-label={label}
        aria-valuetext={formatted}
        title={description ?? `${label}: ${formatted}`}
        className={cn(
          'flex-1 min-w-0 h-1 rounded-full bg-dark-border cursor-pointer',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-dark-accent focus-visible:ring-offset-2 focus-visible:ring-offset-dark-bg',
          accent,
        )}
      />
      <span className="text-dark-text-primary tabular-nums w-16 text-right">
        {formatted}
      </span>
    </label>
  )
}

interface ButtonProps {
  onClick: () => void
  children: ReactNode
  variant?: 'primary' | 'ghost'
  disabled?: boolean
  /** Optional explicit aria-label. Falls back to the string form of children. */
  ariaLabel?: string
  /** Optional type override (defaults to 'button' so we never accidentally submit a form). */
  type?: 'button' | 'submit'
}

export function Button({
  onClick,
  children,
  variant = 'ghost',
  disabled,
  ariaLabel,
  type = 'button',
}: ButtonProps) {
  const fallbackLabel =
    ariaLabel ?? (typeof children === 'string' ? children : undefined)
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      aria-label={fallbackLabel}
      className={cn(
        'px-3 py-1 rounded-md text-[11px] font-mono uppercase tracking-wider transition-all',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-dark-accent focus-visible:ring-offset-2 focus-visible:ring-offset-dark-bg',
        variant === 'primary'
          ? 'bg-dark-accent text-white hover:bg-dark-accent-hover'
          : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary hover:border-dark-border-hover',
        disabled && 'opacity-40 cursor-not-allowed hover:border-dark-border hover:text-dark-text-secondary',
      )}
    >
      {children}
    </button>
  )
}

// A tiny read-only numeric readout — matches the slider value style.
export function Readout({
  label,
  value,
  accent,
}: {
  label: string
  value: string
  accent?: string
}) {
  return (
    <div className="flex items-center gap-2 font-mono text-[11px]">
      <span className="text-dark-text-disabled uppercase tracking-wider">{label}</span>
      <span className={cn('tabular-nums text-dark-text-primary', accent)}>{value}</span>
    </div>
  )
}

// Tiny share button in the widget header — links to the standalone /widget/[name] page.
function ShareLink({ widgetName }: { widgetName: string }) {
  const [copied, setCopied] = useState(false)

  const handleClick = async () => {
    const url = `${window.location.origin}/widget/${widgetName}`
    await navigator.clipboard.writeText(url)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      title={`Share ${widgetName} widget`}
      aria-label={`Copy link to ${widgetName} widget`}
      className="flex items-center gap-1 p-0.5 rounded text-dark-text-disabled hover:text-dark-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-dark-accent transition-colors"
    >
      <Link2 className="w-3 h-3" />
      {copied && <span className="text-[10px]">copied!</span>}
    </button>
  )
}

// Help popover anchored to the widget header. Uses native click to toggle and
// Escape/outside-click to dismiss. Intentionally not a full tooltip library —
// the surface area is tiny and we already depend on lucide-react for icons.
function HelpPopover({ content }: { content: ReactNode }) {
  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const id = useId()

  useEffect(() => {
    if (!open) return
    function onDocClick(e: MouseEvent) {
      if (!containerRef.current) return
      if (!containerRef.current.contains(e.target as Node)) setOpen(false)
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    window.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      window.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <div ref={containerRef} className="relative inline-flex">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-label={open ? 'Close widget help' : 'Open widget help'}
        aria-expanded={open}
        aria-controls={id}
        className="p-0.5 rounded text-dark-text-disabled hover:text-dark-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-dark-accent transition-colors"
      >
        <HelpCircle className="w-3 h-3" />
      </button>
      {open && (
        <div
          id={id}
          role="dialog"
          aria-label="Widget help"
          className="absolute top-6 left-0 z-20 w-72 p-3 rounded-md border border-dark-border bg-dark-surface shadow-xl font-mono text-[11px] leading-relaxed text-dark-text-secondary normal-case tracking-normal"
        >
          {content}
        </div>
      )}
    </div>
  )
}
