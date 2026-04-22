'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout, Slider } from './WidgetFrame'
import { cn } from '@/lib/utils'

// One neuron, spelled out. Three inputs, three weights, one bias. The top
// "anatomy" row shows each multiplication in flight. The middle shows the
// pre-activation sum. The bottom shows the activation output. Change the
// activation and every downstream value updates.

type ActName = 'relu' | 'sigmoid' | 'tanh' | 'linear'

const actLabels: Record<ActName, string> = {
  relu: 'ReLU',
  sigmoid: 'Sigmoid',
  tanh: 'Tanh',
  linear: 'Linear',
}

const actFormulas: Record<ActName, string> = {
  relu: 'max(0, z)',
  sigmoid: '1 / (1 + e⁻ᶻ)',
  tanh: 'tanh(z)',
  linear: 'z (no squash)',
}

function applyAct(act: ActName, z: number): number {
  if (act === 'relu') return Math.max(0, z)
  if (act === 'sigmoid') return 1 / (1 + Math.exp(-z))
  if (act === 'tanh') return Math.tanh(z)
  return z
}

export default function NeuronForward() {
  const [inputs, setInputs] = useState([0.8, -0.5, 1.2])
  const [weights, setWeights] = useState([0.6, -0.4, 0.9])
  const [bias, setBias] = useState(0.1)
  const [act, setAct] = useState<ActName>('relu')

  const products = useMemo(() => inputs.map((x, i) => x * weights[i]), [inputs, weights])
  const z = useMemo(() => products.reduce((s, p) => s + p, 0) + bias, [products, bias])
  const y = applyAct(act, z)

  const setInput = (i: number, v: number) => {
    const next = [...inputs]
    next[i] = v
    setInputs(next)
  }
  const setWeight = (i: number, v: number) => {
    const next = [...weights]
    next[i] = v
    setWeights(next)
  }

  const colors = ['text-term-cyan', 'text-term-pink', 'text-term-amber']

  return (
    <WidgetFrame
      widgetName="NeuronForward"
      label="a neuron in motion"
      right={
        <>
          <span className="font-mono">y = f(Σᵢ wᵢxᵢ + b)</span>
          <span className="text-dark-text-disabled">·</span>
          <span className="font-mono">f = {actFormulas[act]}</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            <span className="text-[11px] font-mono text-dark-text-disabled uppercase tracking-wider mr-1">
              activation
            </span>
            {(['linear', 'relu', 'sigmoid', 'tanh'] as ActName[]).map((a) => (
              <button
                key={a}
                onClick={() => setAct(a)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
                  act === a
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
                )}
              >
                {actLabels[a]}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="z" value={z.toFixed(3)} />
            <Readout label="y = f(z)" value={y.toFixed(4)} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 overflow-auto">
        <div className="flex flex-col gap-5 max-w-[720px] mx-auto">
          {/* Top rail: the three xᵢ · wᵢ products, each with two sliders */}
          <div className="space-y-3">
            {inputs.map((x, i) => (
              <div key={i} className="grid grid-cols-[40px_1fr_20px_1fr_60px_60px] items-center gap-3 font-mono text-[12px]">
                <span className={cn('text-[11px]', colors[i])}>x{sub(i + 1)}</span>
                <div className="h-5 bg-dark-surface-elevated/40 rounded relative">
                  <input
                    type="range"
                    min={-2}
                    max={2}
                    step={0.05}
                    value={x}
                    onChange={(e) => setInput(i, Number(e.target.value))}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                  />
                  <div className="absolute inset-y-0 left-1/2 w-px bg-dark-border" />
                  <div
                    className={cn(
                      'absolute top-0 bottom-0 rounded pointer-events-none',
                      x >= 0 ? 'bg-term-cyan/40' : 'bg-term-rose/40'
                    )}
                    style={{
                      left: `${x >= 0 ? 50 : 50 + (x / 2) * 50}%`,
                      width: `${(Math.abs(x) / 2) * 50}%`,
                    }}
                  />
                  <span className="absolute inset-0 flex items-center justify-end pr-2 text-[10px] tabular-nums pointer-events-none">
                    {x.toFixed(2)}
                  </span>
                </div>
                <span className="text-dark-text-disabled text-center">·</span>
                <div className="h-5 bg-dark-surface-elevated/40 rounded relative">
                  <input
                    type="range"
                    min={-1.5}
                    max={1.5}
                    step={0.05}
                    value={weights[i]}
                    onChange={(e) => setWeight(i, Number(e.target.value))}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                  />
                  <div className="absolute inset-y-0 left-1/2 w-px bg-dark-border" />
                  <div
                    className={cn(
                      'absolute top-0 bottom-0 rounded pointer-events-none',
                      weights[i] >= 0 ? 'bg-term-purple/50' : 'bg-term-rose/40'
                    )}
                    style={{
                      left: `${weights[i] >= 0 ? 50 : 50 + (weights[i] / 1.5) * 50}%`,
                      width: `${(Math.abs(weights[i]) / 1.5) * 50}%`,
                    }}
                  />
                  <span className="absolute inset-0 flex items-center justify-end pr-2 text-[10px] tabular-nums pointer-events-none">
                    w{sub(i + 1)}={weights[i].toFixed(2)}
                  </span>
                </div>
                <span className="text-dark-text-disabled text-right">=</span>
                <span className="text-dark-text-primary tabular-nums text-right font-semibold">
                  {products[i].toFixed(3)}
                </span>
              </div>
            ))}
          </div>

          {/* Sum row */}
          <div className="flex items-center justify-between px-4 py-3 rounded border border-dark-border bg-dark-surface-elevated/30 font-mono">
            <span className="text-[11px] text-dark-text-disabled uppercase tracking-wider">
              pre-activation
            </span>
            <div className="flex items-center gap-2 text-[12px]">
              <span className="text-dark-text-muted">
                {products.map((p, i) => (
                  <span key={i}>
                    {i > 0 && <span className="mx-1">+</span>}
                    <span className={colors[i]}>{p.toFixed(3)}</span>
                  </span>
                ))}
                <span className="mx-1">+</span>
                <span className="text-term-green">
                  {bias.toFixed(2)}
                </span>
              </span>
              <span className="text-dark-text-disabled mx-2">=</span>
              <span className="text-dark-text-primary font-semibold tabular-nums">
                z = {z.toFixed(3)}
              </span>
            </div>
          </div>

          {/* Bias slider */}
          <div className="flex items-center gap-3">
            <Slider
              label="bias b"
              value={bias}
              min={-1.5}
              max={1.5}
              step={0.05}
              onChange={setBias}
              accent="accent-term-green"
            />
          </div>

          {/* Activation arrow */}
          <div className="flex items-center gap-3">
            <span className="text-[11px] font-mono text-dark-text-disabled uppercase tracking-wider w-20">
              activation
            </span>
            <span className="text-dark-accent">→</span>
            <span className="font-mono text-[11px] text-dark-text-muted">
              {actLabels[act]}(z) = {actFormulas[act]}
            </span>
            <span className="ml-auto font-mono text-[12px] text-term-amber font-semibold tabular-nums">
              y = {y.toFixed(4)}
            </span>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function sub(n: number): string {
  return String(n)
    .split('')
    .map((d) => '₀₁₂₃₄₅₆₇₈₉'[Number(d)] ?? d)
    .join('')
}
