'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Button, Readout } from './WidgetFrame'
import { RotateCcw, Play } from 'lucide-react'
import { cn } from '@/lib/utils'

// A tiny single-neuron binary classifier. You pick an input, a target, and a
// learning rate, then step through gradient descent and watch each individual
// gradient and the resulting weight/bias updates. One full step = one forward
// pass + one backward pass + one parameter update. This is the whole training
// loop, stripped of all bookkeeping, laid bare on the page.

// Input + target are fixed for clarity.
const X = [0.8, -0.4, 1.2]
const TARGET = 1.0 // we want sigmoid(z) to go to 1

function sigmoid(z: number) {
  return 1 / (1 + Math.exp(-z))
}

export default function BackpropOneLayer() {
  const [w, setW] = useState<number[]>([0.2, -0.1, 0.3])
  const [b, setB] = useState(0.1)
  const [lr, setLr] = useState(0.5)

  const z = useMemo(() => X.reduce((s, xi, i) => s + xi * w[i], 0) + b, [w, b])
  const yHat = sigmoid(z)
  const loss = 0.5 * (yHat - TARGET) ** 2

  // Backward pass, carefully derived.
  const dL_dyHat = yHat - TARGET // d/dyHat of 0.5(yHat - t)^2
  const dyHat_dz = yHat * (1 - yHat) // σ'(z) = σ(z)(1-σ(z))
  const dL_dz = dL_dyHat * dyHat_dz
  const dL_dw = X.map((xi) => dL_dz * xi)
  const dL_db = dL_dz

  const step = () => {
    setW((prev) => prev.map((wi, i) => wi - lr * dL_dw[i]))
    setB((prev) => prev - lr * dL_db)
  }

  const reset = () => {
    setW([0.2, -0.1, 0.3])
    setB(0.1)
  }

  return (
    <WidgetFrame
      widgetName="BackpropOneLayer"
      label="one full step of backprop + gradient descent"
      right={
        <>
          <span className="font-mono">ŷ = σ(w·x + b)</span>
          <span className="text-dark-text-disabled">·</span>
          <span className="font-mono">L = ½(ŷ − y)²</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="lr α"
            value={lr}
            min={0.05}
            max={2}
            step={0.05}
            onChange={setLr}
            accent="accent-term-purple"
          />
          <Button onClick={step} variant="primary">
            <Play className="w-3 h-3 inline -mt-px mr-1" /> one step
          </Button>
          <Button onClick={reset}>
            <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
          </Button>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="ŷ" value={yHat.toFixed(4)} />
            <Readout label="target" value={TARGET.toFixed(2)} accent="text-term-green" />
            <Readout
              label="loss"
              value={loss.toFixed(4)}
              accent={loss < 0.01 ? 'text-term-green' : 'text-term-amber'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 overflow-auto">
        <div className="max-w-[860px] mx-auto grid grid-cols-2 gap-5 font-mono text-[12px]">
          {/* Forward pass */}
          <div className="rounded border border-term-cyan/40 bg-term-cyan/[0.03] p-4">
            <div className="text-[10px] uppercase tracking-wider text-term-cyan mb-2">
              forward pass
            </div>
            <div className="space-y-1.5">
              {X.map((xi, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-dark-text-muted w-16">x{sub(i + 1)}·w{sub(i + 1)}</span>
                  <span className="text-dark-text-disabled">=</span>
                  <span className="tabular-nums text-dark-text-muted">{xi.toFixed(2)}</span>
                  <span className="text-dark-text-disabled">·</span>
                  <span className="tabular-nums text-term-purple">{w[i].toFixed(3)}</span>
                  <span className="text-dark-text-disabled">=</span>
                  <span className="tabular-nums text-dark-text-primary ml-auto">
                    {(xi * w[i]).toFixed(3)}
                  </span>
                </div>
              ))}
              <div className="border-t border-dark-border pt-1.5 flex items-center gap-2">
                <span className="text-dark-text-muted w-16">z</span>
                <span className="text-dark-text-disabled">=</span>
                <span className="ml-auto tabular-nums text-dark-text-primary">
                  {z.toFixed(4)}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-dark-text-muted w-16">ŷ = σ(z)</span>
                <span className="text-dark-text-disabled">=</span>
                <span className="ml-auto tabular-nums text-term-amber">{yHat.toFixed(4)}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-dark-text-muted w-16">L</span>
                <span className="text-dark-text-disabled">=</span>
                <span className="ml-auto tabular-nums text-term-amber font-semibold">
                  {loss.toFixed(4)}
                </span>
              </div>
            </div>
          </div>

          {/* Backward pass */}
          <div className="rounded border border-term-rose/40 bg-term-rose/[0.03] p-4">
            <div className="text-[10px] uppercase tracking-wider text-term-rose mb-2">
              backward pass
            </div>
            <div className="space-y-1.5">
              <GradRow label="dL/dŷ" expr={`ŷ − y`} value={dL_dyHat.toFixed(4)} />
              <GradRow
                label="dŷ/dz"
                expr={`σ(z)(1−σ(z))`}
                value={dyHat_dz.toFixed(4)}
              />
              <GradRow
                label="dL/dz"
                expr={`(dL/dŷ)(dŷ/dz)`}
                value={dL_dz.toFixed(4)}
                highlight
              />
              {dL_dw.map((g, i) => (
                <GradRow
                  key={i}
                  label={`dL/dw${sub(i + 1)}`}
                  expr={`(dL/dz)·x${sub(i + 1)}`}
                  value={g.toFixed(4)}
                  highlight
                />
              ))}
              <GradRow
                label="dL/db"
                expr={`dL/dz`}
                value={dL_db.toFixed(4)}
                highlight
              />
            </div>
          </div>
        </div>

        {/* Update */}
        <div className="max-w-[860px] mx-auto mt-4 rounded border border-dark-accent/50 bg-dark-accent/[0.04] p-4">
          <div className="text-[10px] uppercase tracking-wider text-dark-accent mb-2">
            update (applied when you hit &quot;one step&quot;)
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 font-mono text-[11.5px]">
            {w.map((wi, i) => (
              <div key={i} className="flex flex-col">
                <span className="text-dark-text-disabled">w{sub(i + 1)}</span>
                <span className="tabular-nums text-dark-text-primary">
                  {wi.toFixed(3)}
                </span>
                <span className="text-dark-accent text-[10px]">
                  − {lr.toFixed(2)} · {dL_dw[i].toFixed(4)} → {(wi - lr * dL_dw[i]).toFixed(3)}
                </span>
              </div>
            ))}
            <div className="flex flex-col">
              <span className="text-dark-text-disabled">b</span>
              <span className="tabular-nums text-dark-text-primary">{b.toFixed(3)}</span>
              <span className="text-dark-accent text-[10px]">
                − {lr.toFixed(2)} · {dL_db.toFixed(4)} → {(b - lr * dL_db).toFixed(3)}
              </span>
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function GradRow({
  label,
  expr,
  value,
  highlight,
}: {
  label: string
  expr: string
  value: string
  highlight?: boolean
}) {
  return (
    <div
      className={cn(
        'flex items-center gap-2 px-1 rounded',
        highlight && 'bg-term-rose/[0.06]'
      )}
    >
      <span className={cn('w-20', highlight ? 'text-term-rose' : 'text-dark-text-muted')}>
        {label}
      </span>
      <span className="text-dark-text-disabled">=</span>
      <span className="text-dark-text-muted text-[11px]">{expr}</span>
      <span className="ml-auto tabular-nums text-dark-text-primary">{value}</span>
    </div>
  )
}

function sub(n: number): string {
  return String(n)
    .split('')
    .map((d) => '₀₁₂₃₄₅₆₇₈₉'[Number(d)] ?? d)
    .join('')
}
