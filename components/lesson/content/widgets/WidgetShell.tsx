'use client'

import { useState } from 'react'
import WidgetFrame, { Slider, Button, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Shared placeholder shell for widgets whose bespoke implementation is in a
// backlog pass. Renders a styled term-panel card with the widget's concept
// described, a small interactive element (slider + readouts), and a honest
// "interactive build coming" note. Keeps the narrative lessons shippable.

interface Props {
  label: string
  right?: string
  description: string
  concept: string
  slider?: { label: string; min: number; max: number; step: number; initial: number; unit?: string }
  readouts?: Array<{ label: string; compute: (v: number) => string; accent?: string }>
  accent?: 'purple' | 'amber' | 'cyan' | 'pink' | 'green'
}

const ACCENT_BG = {
  purple: 'rgba(167, 139, 250, 0.1)',
  amber: 'rgba(251, 191, 36, 0.1)',
  cyan: 'rgba(103, 232, 249, 0.1)',
  pink: 'rgba(244, 114, 182, 0.1)',
  green: 'rgba(74, 222, 128, 0.1)',
}

const ACCENT_TEXT = {
  purple: 'text-term-purple',
  amber: 'text-term-amber',
  cyan: 'text-term-cyan',
  pink: 'text-term-pink',
  green: 'text-term-green',
}

export default function WidgetShell({
  label,
  right,
  description,
  concept,
  slider,
  readouts,
  accent = 'purple',
}: Props) {
  const [v, setV] = useState(slider?.initial ?? 0.5)

  return (
    <WidgetFrame
      label={label}
      right={right ? <span className="font-mono">{right}</span> : undefined}
      aspect="wide"
      controls={
        slider ? (
          <div className="flex flex-wrap items-center gap-4">
            <Slider
              label={slider.label}
              value={v}
              min={slider.min}
              max={slider.max}
              step={slider.step}
              onChange={setV}
              format={(n) => n.toFixed(2) + (slider.unit ?? '')}
              accent={`accent-term-${accent}`}
            />
            <div className="flex items-center gap-4 ml-auto">
              {readouts?.map((r, i) => (
                <Readout
                  key={i}
                  label={r.label}
                  value={r.compute(v)}
                  accent={r.accent ?? ACCENT_TEXT[accent]}
                />
              ))}
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-4">
            <span className="text-[11px] font-mono text-dark-text-muted">
              interactive build queued — this is the widget&apos;s concept card
            </span>
          </div>
        )
      }
    >
      <div className="absolute inset-0 p-6 overflow-auto flex items-center justify-center">
        <div className="max-w-[720px] w-full">
          <div
            className="rounded-lg border border-dark-border p-5"
            style={{ backgroundColor: ACCENT_BG[accent] }}
          >
            <div className={cn('text-[10px] font-mono uppercase tracking-wider mb-2', ACCENT_TEXT[accent])}>
              concept
            </div>
            <p className="font-sans text-[13px] text-dark-text-primary leading-relaxed mb-4">
              {concept}
            </p>
            <div className="text-[10px] font-mono uppercase tracking-wider mb-1 text-dark-text-disabled">
              intended interaction
            </div>
            <p className="font-sans text-[12.5px] text-dark-text-secondary leading-relaxed">
              {description}
            </p>
          </div>
          {slider && (
            <div className="mt-4 rounded border border-dark-border bg-dark-surface-elevated/30 p-4">
              <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
                parameter preview
              </div>
              <div className="relative h-10 bg-dark-bg rounded overflow-hidden">
                <div
                  className="absolute inset-y-0 transition-all"
                  style={{
                    left: 0,
                    width: `${((v - slider.min) / (slider.max - slider.min)) * 100}%`,
                    backgroundColor: ACCENT_BG[accent].replace('0.1', '0.4'),
                  }}
                />
                <div className="absolute inset-0 flex items-center justify-center font-mono text-[12px] text-dark-text-primary tabular-nums">
                  {slider.label} = {v.toFixed(2)}
                  {slider.unit ?? ''}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </WidgetFrame>
  )
}
