'use client'

import { useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Given a damaged 64-neuron layer (25% dead), apply rescue strategies and see
// the recovery. Pure UI — no live training, just preset outcomes per strategy,
// so the reader absorbs the menu of fixes without waiting.

type Strategy = 'lower-lr' | 'leaky-relu' | 'reinit-dead' | 'gelu' | null

const initialAlive = 48 // out of 64 — 25% dead

const outcomes: Record<Exclude<Strategy, null>, {
  label: string
  description: string
  finalAlive: number
  pros: string
  cons: string
  color: string
}> = {
  'lower-lr': {
    label: 'Lower LR · 3×',
    description: 'Reduce the learning rate so the update to biases doesn\'t drive them so negative.',
    finalAlive: 56,
    pros: 'one-line code change, no model surgery',
    cons: 'won\'t revive already-dead neurons; also slows learning',
    color: '#67e8f9',
  },
  'leaky-relu': {
    label: 'Swap → LeakyReLU',
    description: 'Give the negative side a 0.1 slope. Gradients flow through "asleep" neurons so they can recover.',
    finalAlive: 64,
    pros: 'revives every dead neuron; minimal accuracy cost',
    cons: 'slight loss of sparsity; fewer representational advantages of pure ReLU',
    color: '#fbbf24',
  },
  'reinit-dead': {
    label: 'Re-init dead neurons',
    description: 'Detect dead neurons, randomize their weights with fresh He init, continue training.',
    finalAlive: 63,
    pros: 'keeps ReLU; surgical recovery',
    cons: 'requires instrumentation; can disrupt trained parts of the network',
    color: '#a78bfa',
  },
  'gelu': {
    label: 'Swap → GELU',
    description: 'Replace ReLU with GELU (smooth approximation). No hard-zero region, so nothing fully dies.',
    finalAlive: 62,
    pros: 'modern default in transformers; dying is impossible',
    cons: 'slightly more compute per forward; different gradient characteristics',
    color: '#4ade80',
  },
}

export default function RescueMode() {
  const [strategy, setStrategy] = useState<Strategy>(null)
  const currentAlive = strategy ? outcomes[strategy].finalAlive : initialAlive
  const savedCount = currentAlive - initialAlive

  return (
    <WidgetFrame
      widgetName="RescueMode"
      label="rescue mode — fixing a network with 25% dead neurons"
      right={<span className="font-mono">64-neuron ReLU layer · 16 dead on arrival</span>}
      aspect="wide"
      controls={
        <div className="flex items-center gap-4">
          <div className="text-[11px] font-mono text-dark-text-muted">
            click a strategy to see its effect
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="alive"
              value={`${currentAlive} / 64`}
              accent={currentAlive >= 60 ? 'text-term-green' : currentAlive >= 50 ? 'text-term-amber' : 'text-term-rose'}
            />
            {strategy && (
              <Readout
                label="recovered"
                value={`+${savedCount}`}
                accent="text-term-green"
              />
            )}
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 grid grid-cols-1 md:grid-cols-[1fr_300px] gap-4 overflow-auto">
        {/* Neuron grid */}
        <div className="flex flex-col gap-3">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            per-neuron activation rate
          </div>
          <div className="grid grid-cols-8 gap-1 max-w-[360px]">
            {Array.from({ length: 64 }).map((_, i) => {
              let alive = i < initialAlive
              if (strategy === 'lower-lr') alive = i < 56
              if (strategy === 'leaky-relu') alive = true
              if (strategy === 'reinit-dead') alive = i < 63
              if (strategy === 'gelu') alive = i < 62
              return (
                <div
                  key={i}
                  className={cn(
                    'aspect-square rounded-sm transition-all',
                    alive ? 'bg-term-amber/70 ring-1 ring-term-amber/60' : 'bg-dark-bg ring-1 ring-term-rose/40'
                  )}
                />
              )
            })}
          </div>
          <div className="text-[10.5px] font-mono text-dark-text-disabled flex items-center gap-4">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2.5 h-2.5 bg-term-amber/70 rounded-sm" />
              active
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2.5 h-2.5 bg-dark-bg border border-term-rose/50 rounded-sm" />
              dead
            </span>
          </div>

          {strategy && (
            <div className="mt-4 rounded border border-dark-border bg-dark-surface-elevated/30 p-3 text-[12px] font-sans text-dark-text-secondary leading-relaxed">
              <div className="font-mono text-[10px] uppercase tracking-wider mb-1" style={{ color: outcomes[strategy].color }}>
                {outcomes[strategy].label}
              </div>
              <p className="mb-2">{outcomes[strategy].description}</p>
              <div className="grid grid-cols-2 gap-2 text-[11px]">
                <div><span className="text-term-green">pro</span> — {outcomes[strategy].pros}</div>
                <div><span className="text-term-rose">con</span> — {outcomes[strategy].cons}</div>
              </div>
            </div>
          )}
        </div>

        {/* Strategy buttons */}
        <div className="flex flex-col gap-2">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-1">
            choose a rescue
          </div>
          {(Object.keys(outcomes) as Array<keyof typeof outcomes>).map((key) => {
            const o = outcomes[key]
            const active = strategy === key
            return (
              <button
                key={key}
                onClick={() => setStrategy(active ? null : key)}
                className={cn(
                  'text-left px-3 py-2.5 rounded border transition-all',
                  active ? 'bg-dark-accent/10' : 'hover:bg-white/[0.03]'
                )}
                style={active ? { borderColor: o.color } : { borderColor: '#2a2a2a' }}
              >
                <div className="font-mono text-[11.5px] text-dark-text-primary font-semibold">
                  {o.label}
                </div>
                <div className="font-mono text-[10px] text-dark-text-disabled mt-0.5">
                  → {o.finalAlive} / 64 alive
                </div>
              </button>
            )
          })}
        </div>
      </div>
    </WidgetFrame>
  )
}
