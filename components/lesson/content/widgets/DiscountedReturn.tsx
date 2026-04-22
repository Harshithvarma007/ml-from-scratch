'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// A 20-step reward tape with planted spikes. The slider drags γ from 0 to 1.
// Each bar is colored by its weight γ^t; the total discounted return Σ γ^t r_t
// updates live, and a second track shows the naive undiscounted sum so you can
// compare. γ ≈ 0 only respects the next step; γ ≈ 1 treats every reward equally.

const REWARDS: number[] = [
  1, 0, -1, 1, 0, 5, 0, 0, -1, 1,
  0, 0, 5, -1, 1, 0, 0, 1, 0, -1,
]

function effectiveHorizon(g: number): number {
  if (g <= 0.001) return 1
  if (g >= 0.999) return Infinity
  return 1 / (1 - g)
}

export default function DiscountedReturn() {
  const [gamma, setGamma] = useState(0.9)

  const weights = useMemo(() => REWARDS.map((_, t) => Math.pow(gamma, t)), [gamma])
  const terms = useMemo(() => REWARDS.map((r, t) => weights[t] * r), [weights])
  const discounted = terms.reduce((a, v) => a + v, 0)
  const raw = REWARDS.reduce((a, v) => a + v, 0)

  const horizon = effectiveHorizon(gamma)

  // Range for drawing bars — weighted-term min/max with a small pad.
  const termMax = Math.max(...terms.map((v) => Math.abs(v)), 0.5)

  return (
    <WidgetFrame
      widgetName="DiscountedReturn"
      label="discounted return — Σ γ^t · r_t over a 20-step tape"
      right={<span className="font-mono">γ→0 greedy · γ→1 patient</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="γ"
            value={gamma}
            min={0}
            max={0.999}
            step={0.001}
            onChange={setGamma}
            format={(v) => v.toFixed(3)}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="raw Σr" value={raw.toFixed(2)} accent="text-dark-text-primary" />
            <Readout label="G = Σ γ^t r_t" value={discounted.toFixed(3)} accent="text-term-amber" />
            <Readout
              label="horizon 1/(1−γ)"
              value={horizon === Infinity ? '∞' : horizon.toFixed(1)}
              accent="text-term-cyan"
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-rows-[auto_1fr_auto] gap-3 overflow-hidden">
        {/* Header row */}
        <div className="grid grid-cols-[48px_1fr_1fr_1fr_1fr] gap-2 text-[9.5px] font-mono uppercase tracking-wider text-dark-text-disabled px-1">
          <div>t</div>
          <div>r_t</div>
          <div>γ^t (weight)</div>
          <div>γ^t · r_t (term)</div>
          <div className="text-right">running G</div>
        </div>

        {/* Rows */}
        <div className="min-h-0 overflow-auto rounded border border-dark-border bg-dark-bg">
          <div className="grid" style={{ gridTemplateColumns: '48px 1fr 1fr 1fr 1fr' }}>
            {REWARDS.map((r, t) => {
              const w = weights[t]
              const term = terms[t]
              const running = terms.slice(0, t + 1).reduce((a, v) => a + v, 0)
              const rewardColor =
                r >= 5
                  ? 'text-term-green'
                  : r > 0
                  ? 'text-term-emerald'
                  : r < 0
                  ? 'text-term-rose'
                  : 'text-dark-text-muted'
              return (
                <div key={t} className="contents">
                  <div className="px-2 py-1 text-[10.5px] font-mono text-dark-text-muted border-b border-dark-border/40 tabular-nums">
                    {t}
                  </div>
                  <div className="px-2 py-1 border-b border-dark-border/40 flex items-center gap-2">
                    <RewardChip r={r} />
                    <span className={cn('font-mono text-[10.5px] tabular-nums', rewardColor)}>
                      {r > 0 ? '+' : ''}
                      {r}
                    </span>
                  </div>
                  <div className="px-2 py-1 border-b border-dark-border/40 flex items-center gap-2">
                    <div className="flex-1 h-2 bg-dark-surface-elevated/40 rounded-sm overflow-hidden">
                      <div
                        className="h-full bg-term-amber/70"
                        style={{ width: `${w * 100}%` }}
                      />
                    </div>
                    <span className="font-mono text-[10px] tabular-nums text-dark-text-secondary w-12 text-right">
                      {w.toFixed(3)}
                    </span>
                  </div>
                  <div className="px-2 py-1 border-b border-dark-border/40 flex items-center gap-2">
                    <div className="relative flex-1 h-2 bg-dark-surface-elevated/40 rounded-sm overflow-hidden">
                      <div className="absolute top-0 bottom-0 border-l border-dark-border/80" style={{ left: '50%' }} />
                      <div
                        className="absolute top-0 bottom-0"
                        style={{
                          left: term >= 0 ? '50%' : `${50 - (Math.abs(term) / termMax) * 50}%`,
                          width: `${(Math.abs(term) / termMax) * 50}%`,
                          backgroundColor: term >= 0 ? 'rgba(74, 222, 128, 0.8)' : 'rgba(251, 113, 133, 0.8)',
                        }}
                      />
                    </div>
                    <span
                      className={cn(
                        'font-mono text-[10px] tabular-nums w-14 text-right',
                        term > 0.01 ? 'text-term-green' : term < -0.01 ? 'text-term-rose' : 'text-dark-text-muted',
                      )}
                    >
                      {term >= 0 ? '+' : ''}
                      {term.toFixed(3)}
                    </span>
                  </div>
                  <div className="px-2 py-1 border-b border-dark-border/40 text-right font-mono text-[10.5px] tabular-nums text-term-amber">
                    {running.toFixed(3)}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Footer: comparison bar */}
        <div className="grid grid-cols-2 gap-3 px-1">
          <ComparisonBar label="undiscounted" value={raw} span={Math.max(Math.abs(raw), Math.abs(discounted), 1)} color="#888888" />
          <ComparisonBar label="G (γ-discounted)" value={discounted} span={Math.max(Math.abs(raw), Math.abs(discounted), 1)} color="#fbbf24" />
        </div>
      </div>
    </WidgetFrame>
  )
}

function RewardChip({ r }: { r: number }) {
  if (r === 0) return <div className="w-2 h-2 rounded-sm bg-dark-border" />
  const big = Math.abs(r) >= 5
  const color = r > 0 ? (big ? '#4ade80' : '#34d399') : big ? '#fb7185' : '#f87171'
  return (
    <div
      className={cn('w-2 h-2 rounded-sm', big && 'ring-1 ring-offset-1 ring-offset-dark-bg')}
      style={{ backgroundColor: color }}
    />
  )
}

function ComparisonBar({ label, value, span, color }: { label: string; value: number; span: number; color: string }) {
  const pct = Math.min(1, Math.abs(value) / span)
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between font-mono text-[10px]">
        <span className="text-dark-text-disabled uppercase tracking-wider">{label}</span>
        <span className="tabular-nums" style={{ color }}>
          {value.toFixed(3)}
        </span>
      </div>
      <div className="relative h-3 bg-dark-surface-elevated/40 rounded-sm overflow-hidden">
        <div className="absolute top-0 bottom-0 border-l border-dark-border/80" style={{ left: '50%' }} />
        <div
          className="absolute top-0 bottom-0"
          style={{
            left: value >= 0 ? '50%' : `${50 - pct * 50}%`,
            width: `${pct * 50}%`,
            backgroundColor: color,
            opacity: 0.8,
          }}
        />
      </div>
    </div>
  )
}
