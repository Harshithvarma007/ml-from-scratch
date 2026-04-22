'use client'

import { useState } from 'react'
import WidgetFrame, { Readout, Button } from './WidgetFrame'
import { RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'

// Two stacked bar charts — predicted distribution and the one-hot target.
// Pick which class is the true one, then drag the predicted bars up and down.
// The loss readout updates in real time. A perfect match gives 0; a confident
// wrong answer sends it to the moon.

const CLASSES = ['cat', 'dog', 'bird', 'fish', 'fox']
const INIT: number[] = [0.2, 0.2, 0.2, 0.2, 0.2]

function normalize(p: number[]): number[] {
  const s = p.reduce((a, b) => a + b, 0) || 1
  return p.map((v) => v / s)
}

function crossEntropy(p: number[], targetIdx: number): number {
  // H(y, p) = -log p[target]
  const v = p[targetIdx]
  return v > 0 ? -Math.log(v) : Infinity
}

export default function CrossEntropyExplorer() {
  const [probs, setProbs] = useState<number[]>([...INIT])
  const [target, setTarget] = useState(0)

  const setProb = (i: number, v: number) => {
    const next = [...probs]
    next[i] = Math.max(0.001, Math.min(0.999, v))
    setProbs(normalize(next))
  }

  const reset = () => {
    setProbs([...INIT])
    setTarget(0)
  }

  const loss = crossEntropy(probs, target)
  const pTarget = probs[target]

  const lossColor =
    loss < 0.3 ? 'text-term-green' : loss < 1.5 ? 'text-term-amber' : 'text-term-rose'

  return (
    <WidgetFrame
      widgetName="CrossEntropyExplorer"
      label="cross-entropy — the distance from right"
      right={<span className="font-mono">H(y, p) = − log p[target]</span>}
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
                    ? 'bg-term-amber text-dark-bg'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
                )}
              >
                {c}
              </button>
            ))}
          </div>
          <Button onClick={reset}>
            <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
          </Button>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="p(target)" value={pTarget.toFixed(3)} />
            <Readout label="loss" value={loss.toFixed(3)} accent={lossColor} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 grid grid-cols-[1fr_1fr] gap-6 p-5">
        {/* Predicted distribution */}
        <div className="flex flex-col min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
            predicted p
          </div>
          <div className="flex-1 flex flex-col gap-2 justify-center">
            {probs.map((p, i) => (
              <div key={i} className="flex items-center gap-3">
                <span
                  className={cn(
                    'w-10 text-[11px] font-mono',
                    i === target ? 'text-term-amber' : 'text-dark-text-secondary'
                  )}
                >
                  {CLASSES[i]}
                </span>
                <div className="flex-1 relative h-5 bg-dark-surface-elevated/40 rounded overflow-hidden">
                  <input
                    type="range"
                    min={0.001}
                    max={0.999}
                    step={0.005}
                    value={p}
                    onChange={(e) => setProb(i, Number(e.target.value))}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                  />
                  <div
                    className={cn(
                      'absolute inset-y-0 left-0 transition-all pointer-events-none',
                      i === target ? 'bg-term-amber/60' : 'bg-term-purple/40'
                    )}
                    style={{ width: `${p * 100}%` }}
                  />
                </div>
                <span
                  className={cn(
                    'w-12 text-right font-mono text-[10.5px] tabular-nums',
                    i === target ? 'text-term-amber' : 'text-dark-text-muted'
                  )}
                >
                  {(p * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
          <div className="text-[10px] font-mono text-dark-text-disabled mt-2">
            drag any row · values auto-normalise
          </div>
        </div>

        {/* Target distribution (one-hot) */}
        <div className="flex flex-col min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
            target y (one-hot)
          </div>
          <div className="flex-1 flex flex-col gap-2 justify-center">
            {CLASSES.map((c, i) => (
              <div key={c} className="flex items-center gap-3">
                <span
                  className={cn(
                    'w-10 text-[11px] font-mono',
                    i === target ? 'text-term-green' : 'text-dark-text-disabled'
                  )}
                >
                  {c}
                </span>
                <div className="flex-1 h-5 bg-dark-surface-elevated/40 rounded overflow-hidden">
                  <div
                    className={cn(
                      'h-full transition-all',
                      i === target ? 'bg-term-green/70' : 'bg-transparent'
                    )}
                    style={{ width: `${i === target ? 100 : 0}%` }}
                  />
                </div>
                <span
                  className={cn(
                    'w-12 text-right font-mono text-[10.5px] tabular-nums',
                    i === target ? 'text-term-green' : 'text-dark-text-disabled'
                  )}
                >
                  {i === target ? '100%' : '0%'}
                </span>
              </div>
            ))}
          </div>
          <div className="text-[10px] font-mono text-dark-text-disabled mt-2">
            exactly one class is correct
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
