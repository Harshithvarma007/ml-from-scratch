'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// A 10-step trajectory with hand-crafted rewards. For each step t we compute
// the return-to-go G_t = Σ_{k≥t} γ^{k−t} r_k, then render the gradient
// contribution ∇log π(a_t|s_t) · G_t as a bar whose size and sign are
// driven entirely by G_t (we show the log-prob magnitude as a constant-ish
// "nudge", so the visual emphasis is on the return scaling).

const T = 10
// Synthesized trajectory: a failed start, then three good picks, one bad,
// then a payoff at the end.
const REWARDS = [0, -1, 0, +2, +1, 0, -2, +1, 0, +5]
// Pretend log-probs (magnitudes of the policy gradient direction for each step).
// These are the |∇log π(a_t|s_t)| used for visualization only.
const LOG_PROB_MAG = [0.9, 1.1, 0.6, 1.4, 1.2, 0.8, 1.5, 0.9, 0.7, 1.3]

function returnsToGo(rewards: number[], gamma: number): number[] {
  const G: number[] = new Array(rewards.length).fill(0)
  let acc = 0
  for (let t = rewards.length - 1; t >= 0; t--) {
    acc = rewards[t] + gamma * acc
    G[t] = acc
  }
  return G
}

export default function ReturnToGradient() {
  const [gamma, setGamma] = useState(0.9)
  const G = useMemo(() => returnsToGo(REWARDS, gamma), [gamma])
  const grads = G.map((g, t) => g * LOG_PROB_MAG[t])
  const gMax = Math.max(...G.map(Math.abs), 1e-3)
  const gradMax = Math.max(...grads.map(Math.abs), 1e-3)

  const totalPos = grads.filter((v) => v > 0).reduce((a, v) => a + v, 0)
  const totalNeg = grads.filter((v) => v < 0).reduce((a, v) => a + v, 0)

  return (
    <WidgetFrame
      widgetName="ReturnToGradient"
      label="return-to-go → gradient weight"
      right={<span className="font-mono">∇θ log π(a_t|s_t) · G_t</span>}
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
            <Readout label="Σ grad⁺" value={totalPos.toFixed(2)} accent="text-term-green" />
            <Readout label="Σ grad⁻" value={totalNeg.toFixed(2)} accent="text-term-rose" />
            <Readout label="net" value={(totalPos + totalNeg).toFixed(2)} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-auto">
        <div className="grid grid-cols-[46px_1fr_1fr_1fr_1fr] gap-2 text-[9.5px] font-mono uppercase tracking-wider text-dark-text-disabled px-1 mb-1">
          <div>t</div>
          <div>r_t</div>
          <div>G_t (return-to-go)</div>
          <div>|∇log π|</div>
          <div>∇log π · G_t</div>
        </div>

        <div className="rounded border border-dark-border bg-dark-bg divide-y divide-dark-border/40">
          {REWARDS.map((r, t) => {
            const g = G[t]
            const grad = grads[t]
            return (
              <div
                key={t}
                className="grid gap-2 px-1 py-1.5"
                style={{ gridTemplateColumns: '46px 1fr 1fr 1fr 1fr' }}
              >
                <div className="text-[10.5px] font-mono text-dark-text-muted tabular-nums self-center pl-2">
                  {t}
                </div>
                {/* Reward chip */}
                <div className="flex items-center gap-2">
                  <RewardDot r={r} />
                  <span
                    className={cn(
                      'font-mono text-[10.5px] tabular-nums',
                      r > 0 ? 'text-term-green' : r < 0 ? 'text-term-rose' : 'text-dark-text-muted',
                    )}
                  >
                    {r > 0 ? '+' : ''}
                    {r}
                  </span>
                </div>
                {/* G_t bar */}
                <BarCell value={g} max={gMax} color="#fbbf24" />
                {/* |grad| bar */}
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-2.5 bg-dark-surface-elevated/40 rounded-sm overflow-hidden">
                    <div
                      className="h-full bg-term-purple/70"
                      style={{ width: `${(LOG_PROB_MAG[t] / 1.5) * 100}%` }}
                    />
                  </div>
                  <span className="font-mono text-[10px] tabular-nums text-term-purple w-10 text-right">
                    {LOG_PROB_MAG[t].toFixed(2)}
                  </span>
                </div>
                {/* Grad contribution bar */}
                <BarCell value={grad} max={gradMax} color={grad >= 0 ? '#4ade80' : '#fb7185'} highlight />
              </div>
            )
          })}
        </div>

        <div className="mt-3 text-[11px] font-mono text-dark-text-muted leading-snug">
          positive G_t pushes log π(a_t|s_t) up — the network makes that action more likely. negative G_t does
          the opposite. steps that land near a big future reward get amplified, while unrelated steps shrink
          toward zero.
        </div>
      </div>
    </WidgetFrame>
  )
}

function RewardDot({ r }: { r: number }) {
  if (r === 0) return <div className="w-2 h-2 rounded-sm bg-dark-border" />
  const big = Math.abs(r) >= 3
  return (
    <div
      className={cn('w-2 h-2 rounded-sm', big && 'ring-1 ring-offset-1 ring-offset-dark-bg')}
      style={{
        backgroundColor: r > 0 ? (big ? '#4ade80' : '#34d399') : big ? '#fb7185' : '#f87171',
      }}
    />
  )
}

function BarCell({
  value,
  max,
  color,
  highlight,
}: {
  value: number
  max: number
  color: string
  highlight?: boolean
}) {
  const pct = Math.min(1, Math.abs(value) / max)
  return (
    <div className="flex items-center gap-2">
      <div
        className={cn(
          'relative flex-1 h-2.5 bg-dark-surface-elevated/40 rounded-sm overflow-hidden',
          highlight && 'ring-1 ring-dark-border/70',
        )}
      >
        <div className="absolute top-0 bottom-0 border-l border-dark-border/80" style={{ left: '50%' }} />
        <div
          className="absolute top-0 bottom-0"
          style={{
            left: value >= 0 ? '50%' : `${50 - pct * 50}%`,
            width: `${pct * 50}%`,
            backgroundColor: color,
            opacity: 0.85,
          }}
        />
      </div>
      <span
        className="font-mono text-[10px] tabular-nums w-14 text-right"
        style={{ color: value >= 0 ? color : color }}
      >
        {value >= 0 ? '+' : ''}
        {value.toFixed(2)}
      </span>
    </div>
  )
}
