'use client'

import { useState } from 'react'
import WidgetFrame, { Button } from './WidgetFrame'
import { cn } from '@/lib/utils'
import { StepForward, StepBack, SkipBack } from 'lucide-react'

// Six boxed equations that walk from the objective J(θ) to the REINFORCE
// estimator. Clicking "step" advances. The most recent box glows amber; the
// "what changed" callout highlights the transformation we just applied
// (log-derivative trick, monte-carlo swap, etc.).

type Eq = {
  lhs: string
  rhs: string
  label: string
  change: string
}

const EQS: Eq[] = [
  {
    label: '1 · objective',
    lhs: 'J(θ)',
    rhs: '= E_{τ ~ π_θ} [ R(τ) ]',
    change:
      'goal: pick θ that maximises expected return over trajectories sampled from our policy.',
  },
  {
    label: '2 · expand expectation',
    lhs: '∇_θ J(θ)',
    rhs: '= ∇_θ ∫ π_θ(τ) · R(τ) dτ',
    change:
      'swap the E for an explicit integral over trajectories — now the gradient can enter.',
  },
  {
    label: '3 · push gradient inside',
    lhs: '∇_θ J(θ)',
    rhs: '= ∫ ∇_θ π_θ(τ) · R(τ) dτ',
    change:
      'gradient moves inside the integral. R(τ) is constant in θ, so only π_θ(τ) gets differentiated.',
  },
  {
    label: '4 · log-derivative trick',
    lhs: '∇_θ π_θ(τ)',
    rhs: '= π_θ(τ) · ∇_θ log π_θ(τ)',
    change:
      'the identity ∇f = f · ∇log f — multiply and divide by π_θ(τ) to recover an expectation.',
  },
  {
    label: '5 · expectation form',
    lhs: '∇_θ J(θ)',
    rhs: '= E_{τ ~ π_θ} [ ∇_θ log π_θ(τ) · R(τ) ]',
    change:
      'ta-da — a clean expectation. No environment dynamics p(s\'|s,a) sit in ∇_θ log π_θ(τ): they cancel.',
  },
  {
    label: '6 · REINFORCE estimator',
    lhs: '∇̂_θ J(θ)',
    rhs: '≈ (1/N) Σᵢ Σ_t ∇_θ log π_θ(a_t | s_t) · R(τᵢ)',
    change:
      'replace the expectation with a Monte-Carlo average over N sampled trajectories. This is REINFORCE.',
  },
]

export default function PolicyGradientDerivation() {
  const [i, setI] = useState(0)
  const current = EQS[i]

  return (
    <WidgetFrame
      widgetName="PolicyGradientDerivation"
      label="policy gradient — derivation, one equation at a time"
      right={<span className="font-mono">log-derivative trick · Monte-Carlo estimator</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <Button onClick={() => setI(0)}>
              <span className="inline-flex items-center gap-1">
                <SkipBack size={11} /> reset
              </span>
            </Button>
            <Button onClick={() => setI(Math.max(0, i - 1))} disabled={i === 0}>
              <span className="inline-flex items-center gap-1">
                <StepBack size={11} /> back
              </span>
            </Button>
            <Button onClick={() => setI(Math.min(EQS.length - 1, i + 1))} variant="primary" disabled={i === EQS.length - 1}>
              <span className="inline-flex items-center gap-1">
                step <StepForward size={11} />
              </span>
            </Button>
          </div>
          <div className="flex items-center gap-4 ml-auto font-mono text-[11px] text-dark-text-secondary">
            step {i + 1} / {EQS.length}
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-auto">
        <div className="flex flex-col gap-3">
          {EQS.map((eq, idx) => {
            const visible = idx <= i
            const isCurrent = idx === i
            return (
              <div
                key={idx}
                className={cn(
                  'rounded border transition-all',
                  visible
                    ? isCurrent
                      ? 'border-term-amber bg-term-amber/5 shadow-[0_0_12px_rgba(251,191,36,0.15)]'
                      : 'border-dark-border bg-dark-surface-elevated/30'
                    : 'border-dark-border/40 bg-dark-bg opacity-40',
                )}
              >
                <div className="px-4 py-2 flex items-center justify-between border-b border-dark-border/60">
                  <span
                    className={cn(
                      'text-[10.5px] font-mono uppercase tracking-wider',
                      isCurrent ? 'text-term-amber' : 'text-dark-text-disabled',
                    )}
                  >
                    {eq.label}
                  </span>
                  <span className="text-[9.5px] font-mono text-dark-text-disabled">
                    #{idx + 1}
                  </span>
                </div>
                <div className="px-5 py-3.5 flex items-center gap-3 font-mono text-[13.5px] tabular-nums">
                  <span className={cn(isCurrent ? 'text-term-amber' : 'text-dark-text-primary')}>
                    {eq.lhs}
                  </span>
                  <span className="text-dark-text-muted">{eq.rhs}</span>
                </div>
                {visible && (
                  <div
                    className={cn(
                      'px-5 py-2.5 text-[11.5px] font-sans leading-relaxed border-t',
                      isCurrent
                        ? 'border-term-amber/30 text-dark-text-primary'
                        : 'border-dark-border/60 text-dark-text-muted',
                    )}
                  >
                    <span
                      className={cn(
                        'font-mono uppercase tracking-wider text-[9.5px] mr-2',
                        isCurrent ? 'text-term-amber' : 'text-dark-text-disabled',
                      )}
                    >
                      what changed
                    </span>
                    {eq.change}
                  </div>
                )}
              </div>
            )
          })}

          {i === EQS.length - 1 && (
            <div className="rounded border border-term-green/40 bg-term-green/5 px-4 py-3 text-[11.5px] font-mono text-term-green">
              final REINFORCE update: θ ← θ + α · ∇_θ log π_θ(a_t | s_t) · R(τ)
            </div>
          )}
        </div>
      </div>
    </WidgetFrame>
  )
}
