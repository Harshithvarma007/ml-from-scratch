'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Show the miraculous identity: ∂L/∂zᵢ = pᵢ − yᵢ, when L is cross-entropy of
// softmax outputs. Three synced rows — logits z, probs p, gradient p - y. Drag
// the logits and watch the gradient bar.

const CLASSES = ['cat', 'dog', 'bird', 'fish', 'fox']
const INIT = [1.8, 0.9, 0.2, -0.5, -1.2]

function softmax(z: number[]): number[] {
  const m = Math.max(...z)
  const e = z.map((v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

export default function SoftmaxCEGradient() {
  const [logits, setLogits] = useState<number[]>([...INIT])
  const [target, setTarget] = useState(0)

  const probs = useMemo(() => softmax(logits), [logits])
  const grad = probs.map((p, i) => p - (i === target ? 1 : 0))
  const loss = -Math.log(Math.max(1e-12, probs[target]))
  const maxAbsGrad = Math.max(...grad.map(Math.abs), 0.1)

  const setLogit = (i: number, v: number) => {
    const next = [...logits]
    next[i] = v
    setLogits(next)
  }

  return (
    <WidgetFrame
      widgetName="SoftmaxCEGradient"
      label="softmax + cross-entropy = the prettiest gradient in ML"
      right={<span className="font-mono">∂L/∂zᵢ = pᵢ − yᵢ</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            <span className="text-[11px] font-mono text-dark-text-disabled uppercase tracking-wider mr-1">
              target
            </span>
            {CLASSES.map((c, i) => (
              <button
                key={c}
                onClick={() => setTarget(i)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
                  target === i
                    ? 'bg-term-green text-dark-bg'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
                )}
              >
                {c}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="loss" value={loss.toFixed(3)} accent="text-term-amber" />
            <Readout
              label="‖grad‖"
              value={Math.hypot(...grad).toFixed(3)}
              accent="text-term-purple"
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 font-mono text-[11px] overflow-auto">
        <div className="grid grid-cols-[80px_1fr_1fr_1fr] gap-x-4 gap-y-2">
          <div className="text-dark-text-disabled uppercase text-[10px]"></div>
          <div className="text-dark-text-disabled uppercase text-[10px]">logit z</div>
          <div className="text-dark-text-disabled uppercase text-[10px]">p = softmax(z)</div>
          <div className="text-dark-text-disabled uppercase text-[10px]">
            gradient  p − y
          </div>

          {CLASSES.map((c, i) => {
            const z = logits[i]
            const p = probs[i]
            const g = grad[i]
            const isTarget = i === target
            return (
              <div key={c} className="contents">
                <div
                  className={cn(
                    'flex items-center',
                    isTarget ? 'text-term-green' : 'text-dark-text-secondary'
                  )}
                >
                  {c}
                  {isTarget && <span className="ml-1 text-[9px]">• true</span>}
                </div>

                {/* Logit slider bar */}
                <div className="relative h-5 bg-dark-surface-elevated/40 rounded overflow-hidden">
                  <input
                    type="range"
                    min={-3}
                    max={3}
                    step={0.05}
                    value={z}
                    onChange={(e) => setLogit(i, Number(e.target.value))}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                  />
                  <div className="absolute inset-y-0 left-1/2 w-px bg-dark-border pointer-events-none" />
                  <div
                    className={cn(
                      'absolute top-0 bottom-0 transition-all pointer-events-none',
                      z >= 0 ? 'bg-term-purple/60' : 'bg-term-rose/50'
                    )}
                    style={{
                      left: `${z >= 0 ? 50 : 50 + (z / 3) * 50}%`,
                      width: `${(Math.abs(z) / 3) * 50}%`,
                    }}
                  />
                  <span className="absolute inset-0 flex items-center justify-end pr-2 text-[10px] tabular-nums text-dark-text-primary pointer-events-none">
                    {z.toFixed(2)}
                  </span>
                </div>

                {/* Probability bar */}
                <div className="h-5 bg-dark-surface-elevated/40 rounded overflow-hidden relative">
                  <div
                    className={cn(
                      'h-full transition-all',
                      isTarget ? 'bg-term-green/60' : 'bg-term-amber/40'
                    )}
                    style={{ width: `${p * 100}%` }}
                  />
                  <span className="absolute inset-0 flex items-center justify-end pr-2 text-[10px] tabular-nums text-dark-text-primary">
                    {(p * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Gradient bar (centered, left = negative, right = positive) */}
                <div className="relative h-5 bg-dark-surface-elevated/40 rounded overflow-hidden">
                  <div className="absolute inset-y-0 left-1/2 w-px bg-dark-border" />
                  <div
                    className={cn(
                      'absolute top-0 bottom-0 transition-all',
                      g >= 0 ? 'bg-term-rose/60' : 'bg-term-green/60'
                    )}
                    style={{
                      left: `${g >= 0 ? 50 : 50 + (g / maxAbsGrad) * 50}%`,
                      width: `${(Math.abs(g) / maxAbsGrad) * 50}%`,
                    }}
                  />
                  <span className="absolute inset-0 flex items-center justify-end pr-2 text-[10px] tabular-nums text-dark-text-primary">
                    {g >= 0 ? '+' : ''}
                    {g.toFixed(3)}
                  </span>
                </div>
              </div>
            )
          })}
        </div>

        <div className="mt-5 text-[10.5px] text-dark-text-muted leading-relaxed">
          Rose bars push the logit <em>down</em>. Green bars push it <em>up</em>. The true
          class always gets a green push. Every other class gets a rose push proportional to
          how much probability mass it stole.
        </div>
      </div>
    </WidgetFrame>
  )
}
