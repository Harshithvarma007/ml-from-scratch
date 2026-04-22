'use client'

import WidgetFrame from './WidgetFrame'
import { cn } from '@/lib/utils'

// Side-by-side PPO vs TRPO on four axes. Values are stylised benchmark numbers
// drawn from the original TRPO and PPO papers (and standard reference impls);
// exact magnitudes differ across tasks, but the relative picture is stable.
// Each axis picks its own "winner" — we annotate which side actually wins.

type Metric = {
  label: string
  unit: string
  ppo: number
  trpo: number
  winner: 'ppo' | 'trpo'
  winDirection: 'high' | 'low'
  notes: string
}

const METRICS: Metric[] = [
  {
    label: 'wall-clock / iteration',
    unit: 'ms/iter',
    ppo: 120,
    trpo: 380,
    winner: 'ppo',
    winDirection: 'low',
    notes:
      'TRPO computes a Fisher-vector product (conjugate gradient) and runs a backtracking line search every iteration — expensive.',
  },
  {
    label: 'KL constraint enforcement',
    unit: 'KL bound',
    ppo: 0.4,
    trpo: 1.0,
    winner: 'trpo',
    winDirection: 'high',
    notes:
      'TRPO hard-enforces KL(π_new ‖ π_old) ≤ δ via line search. PPO uses a soft surrogate (clipping) that only approximates the trust region.',
  },
  {
    label: 'implementation complexity',
    unit: 'LoC',
    ppo: 120,
    trpo: 550,
    winner: 'ppo',
    winDirection: 'low',
    notes:
      'PPO is ≈100 lines: compute advantages, clip the ratio, take SGD steps. TRPO needs Fisher-vector product, conjugate gradient, line search, Hessian-vector wrap.',
  },
  {
    label: 'sample efficiency',
    unit: 'returns/1M steps',
    ppo: 0.92,
    trpo: 0.95,
    winner: 'trpo',
    winDirection: 'high',
    notes:
      'On continuous-control benchmarks TRPO and PPO are roughly a wash. TRPO edges out on a few hard tasks; PPO wins on most when tuned.',
  },
]

export default function PPOvsTRPO() {
  return (
    <WidgetFrame
      widgetName="PPOvsTRPO"
      label="PPO vs TRPO — what clipping bought you"
      right={<span className="font-mono">4 axes · numbers stylised from Schulman 2015/2017</span>}
      aspect="wide"
    >
      <div className="absolute inset-0 p-4 grid grid-cols-[1.15fr_1fr] gap-4 overflow-hidden">
        <div className="flex flex-col gap-3 min-h-0 min-w-0 overflow-auto">
          {METRICS.map((m) => (
            <MetricRow key={m.label} metric={m} />
          ))}
        </div>

        {/* Right: summary panel */}
        <div className="flex flex-col gap-3 min-h-0 min-w-0 overflow-auto">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            algorithmic contract
          </div>
          <div className="rounded border border-term-cyan/40 bg-term-cyan/5 p-3 text-[11.5px] font-mono leading-relaxed">
            <div className="text-term-cyan font-semibold mb-1">TRPO</div>
            <p className="text-dark-text-primary">
              max_θ E[ rθ · A ]  subject to  KL(πθ ‖ π_old) ≤ δ
            </p>
            <ul className="mt-2 pl-3 list-disc text-dark-text-muted space-y-0.5">
              <li>solve with conjugate gradient + Fisher-vector product</li>
              <li>backtracking line search to satisfy constraint</li>
              <li>exact trust region, hard KL bound</li>
            </ul>
          </div>
          <div className="rounded border border-term-amber/40 bg-term-amber/5 p-3 text-[11.5px] font-mono leading-relaxed">
            <div className="text-term-amber font-semibold mb-1">PPO</div>
            <p className="text-dark-text-primary">
              max_θ E[ min(rθ · A, clip(rθ, 1−ε, 1+ε) · A) ]
            </p>
            <ul className="mt-2 pl-3 list-disc text-dark-text-muted space-y-0.5">
              <li>plain SGD / Adam, no second-order anything</li>
              <li>clipping makes KL soft but bounded in practice</li>
              <li>one-line objective, ~100 LoC end-to-end</li>
            </ul>
          </div>

          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-1">
            why PPO won in practice
          </div>
          <p className="font-sans text-[12px] text-dark-text-muted leading-relaxed">
            PPO keeps most of TRPO's stability while dropping the machinery. The clip does a first-order
            approximation of "stay inside a trust region" without computing Hessians. You pay a small price
            in worst-case KL control; you gain a 3× speedup and a massively simpler codebase.
          </p>
        </div>
      </div>
    </WidgetFrame>
  )
}

function MetricRow({ metric }: { metric: Metric }) {
  const max = Math.max(metric.ppo, metric.trpo)
  const ppoWins = metric.winner === 'ppo'
  return (
    <div className="rounded border border-dark-border bg-dark-surface-elevated/30 p-3">
      <div className="flex items-center justify-between font-mono text-[11px] mb-2">
        <span className="text-dark-text-primary uppercase tracking-wider text-[10px]">
          {metric.label}
        </span>
        <span className="text-dark-text-disabled text-[10px]">
          {metric.unit} · {metric.winDirection === 'low' ? '↓ lower' : '↑ higher'} wins
        </span>
      </div>
      <Bar
        name="PPO"
        color="#fbbf24"
        value={metric.ppo}
        frac={metric.ppo / max}
        winner={ppoWins}
      />
      <Bar
        name="TRPO"
        color="#67e8f9"
        value={metric.trpo}
        frac={metric.trpo / max}
        winner={!ppoWins}
      />
      <div className="mt-2 text-[10.5px] font-mono text-dark-text-muted leading-snug">
        {metric.notes}
      </div>
    </div>
  )
}

function Bar({
  name,
  color,
  value,
  frac,
  winner,
}: {
  name: string
  color: string
  value: number
  frac: number
  winner: boolean
}) {
  return (
    <div className="flex items-center gap-2 font-mono text-[11px] mb-1">
      <span className="w-10" style={{ color }}>
        {name}
      </span>
      <div className="flex-1 h-4 bg-dark-surface-elevated/40 rounded-sm overflow-hidden">
        <div
          className="h-full"
          style={{
            width: `${Math.min(100, frac * 100)}%`,
            backgroundColor: color,
            opacity: winner ? 0.9 : 0.5,
          }}
        />
      </div>
      <span
        className={cn('w-20 text-right tabular-nums', winner && 'font-semibold')}
        style={{ color: winner ? color : '#888' }}
      >
        {value < 1 ? value.toFixed(2) : Math.round(value).toLocaleString('en-US')}
      </span>
      {winner && (
        <span
          className="text-[9px] font-mono uppercase tracking-wider px-1 rounded-sm"
          style={{ color: '#0a0a0a', backgroundColor: color }}
        >
          win
        </span>
      )}
    </div>
  )
}
