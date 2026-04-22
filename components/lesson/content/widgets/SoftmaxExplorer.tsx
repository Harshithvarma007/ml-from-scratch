'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Button, Readout } from './WidgetFrame'
import { RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'

// Drag the logits, watch the probability distribution reshape itself. The two
// side-by-side bar panels (logits, probs) make the softmax look like exactly
// what it is — a smooth warp that turns a list of real numbers into a list of
// numbers that sum to one.

const CLASSES = ['cat', 'dog', 'bird', 'fish', 'fox']
const DEFAULT_LOGITS = [2.2, 1.7, 0.8, -0.1, -1.4]

function softmax(z: number[], temperature: number): number[] {
  const t = Math.max(1e-6, temperature)
  const scaled = z.map((v) => v / t)
  const m = Math.max(...scaled)
  const exps = scaled.map((v) => Math.exp(v - m))
  const s = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / s)
}

function entropy(p: number[]): number {
  let h = 0
  for (const v of p) if (v > 0) h -= v * Math.log2(v)
  return h
}

export default function SoftmaxExplorer() {
  const [logits, setLogits] = useState<number[]>([...DEFAULT_LOGITS])
  const [T, setT] = useState(1.0)

  const probs = useMemo(() => softmax(logits, T), [logits, T])
  const H = useMemo(() => entropy(probs), [probs])
  const argmax = probs.indexOf(Math.max(...probs))
  const maxLogitAbs = Math.max(...logits.map(Math.abs), 3)

  const setLogit = (i: number, v: number) => {
    const next = [...logits]
    next[i] = v
    setLogits(next)
  }

  const reset = () => {
    setLogits([...DEFAULT_LOGITS])
    setT(1.0)
  }

  return (
    <WidgetFrame
      widgetName="SoftmaxExplorer"
      label="softmax explorer"
      right={
        <>
          <span className="font-mono">softmax(z)ᵢ = e^(zᵢ/T) / Σⱼ e^(zⱼ/T)</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="T"
            value={T}
            min={0.1}
            max={5}
            step={0.01}
            onChange={setT}
            accent="accent-term-amber"
          />
          <Button onClick={reset}>
            <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
          </Button>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="argmax" value={CLASSES[argmax]} accent="text-term-amber" />
            <Readout label="entropy" value={H.toFixed(3) + ' bits'} />
            <Readout label="Σp" value={probs.reduce((a, b) => a + b, 0).toFixed(3)} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 grid grid-cols-2 gap-4 p-5">
        {/* Logits column */}
        <div className="flex flex-col min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
            logits z
          </div>
          <div className="flex-1 flex flex-col gap-2 justify-center">
            {logits.map((v, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className="w-10 text-[11px] font-mono text-dark-text-secondary">
                  {CLASSES[i]}
                </span>
                <div className="flex-1 relative h-6 bg-dark-surface-elevated/40 rounded">
                  <input
                    type="range"
                    min={-3}
                    max={3}
                    step={0.05}
                    value={v}
                    onChange={(e) => setLogit(i, Number(e.target.value))}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                  />
                  <div
                    className={cn(
                      'absolute top-0 bottom-0 rounded transition-all',
                      v >= 0 ? 'bg-term-purple/60' : 'bg-term-rose/50'
                    )}
                    style={{
                      left: `${v >= 0 ? 50 : 50 + (v / maxLogitAbs) * 50}%`,
                      width: `${(Math.abs(v) / maxLogitAbs) * 50}%`,
                    }}
                  />
                  <div className="absolute inset-0 border-l border-dark-border/80 left-1/2 pointer-events-none" />
                </div>
                <span className="w-12 text-right font-mono text-[11px] tabular-nums text-dark-text-primary">
                  {v.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
          <div className="text-[10px] font-mono text-dark-text-disabled mt-2">
            drag rows to change logits · purple = positive · rose = negative
          </div>
        </div>

        {/* Probabilities column */}
        <div className="flex flex-col min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
            probabilities p = softmax(z)
          </div>
          <div className="flex-1 flex flex-col gap-2 justify-center">
            {probs.map((p, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className="w-10 text-[11px] font-mono text-dark-text-secondary">
                  {CLASSES[i]}
                </span>
                <div className="flex-1 h-6 bg-dark-surface-elevated/40 rounded overflow-hidden relative">
                  <div
                    className={cn(
                      'absolute inset-y-0 left-0 transition-all',
                      i === argmax ? 'bg-term-amber/70' : 'bg-term-amber/30'
                    )}
                    style={{ width: `${p * 100}%` }}
                  />
                </div>
                <span
                  className={cn(
                    'w-12 text-right font-mono text-[11px] tabular-nums',
                    i === argmax ? 'text-term-amber' : 'text-dark-text-muted'
                  )}
                >
                  {(p * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
          <div className="text-[10px] font-mono text-dark-text-disabled mt-2">
            sums to 1 by construction · temperature warps the sharpness
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
