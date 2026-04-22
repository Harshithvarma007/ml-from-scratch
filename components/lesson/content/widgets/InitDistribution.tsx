'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Pick a layer depth. For each init strategy, draw the histogram of
// activations at that layer. Naive either goes one-hot-ish (saturated) or
// collapses to a spike at zero; He spreads like a proper Gaussian.

type Init = 'naive' | 'xavier' | 'he'
type Act = 'tanh' | 'relu'

const WIDTH = 64
const N_SAMPLES = 500

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gauss(rng: () => number): number {
  const u = Math.max(rng(), 1e-9)
  const v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function simulate(act: Act, init: Init, depth: number): number[] {
  const rng = mulberry32(7)
  const scales: Record<Init, number> = {
    naive: 1,
    xavier: Math.sqrt(1 / WIDTH),
    he: Math.sqrt(2 / WIDTH),
  }
  const scale = scales[init]

  let a: number[][] = []
  for (let i = 0; i < N_SAMPLES; i++) {
    const row: number[] = []
    for (let j = 0; j < WIDTH; j++) row.push(gauss(rng))
    a.push(row)
  }
  for (let layer = 0; layer < depth; layer++) {
    const W: number[][] = []
    for (let i = 0; i < WIDTH; i++) {
      const row: number[] = []
      for (let j = 0; j < WIDTH; j++) row.push(gauss(rng) * scale)
      W.push(row)
    }
    const next: number[][] = []
    for (let s = 0; s < N_SAMPLES; s++) {
      const out: number[] = []
      for (let i = 0; i < WIDTH; i++) {
        let z = 0
        for (let j = 0; j < WIDTH; j++) z += W[i][j] * a[s][j]
        out.push(act === 'relu' ? Math.max(0, z) : Math.tanh(z))
      }
      next.push(out)
    }
    a = next
  }
  return a.flat()
}

function histogram(values: number[], bins: number, lo: number, hi: number): number[] {
  const out = new Array(bins).fill(0)
  for (const v of values) {
    const idx = Math.min(
      bins - 1,
      Math.max(0, Math.floor(((v - lo) / (hi - lo)) * bins)),
    )
    out[idx]++
  }
  return out
}

export default function InitDistribution() {
  const [depth, setDepth] = useState(10)
  const [activation, setActivation] = useState<Act>('relu')

  const dists = useMemo(
    () =>
      ({
        naive: simulate(activation, 'naive', depth),
        xavier: simulate(activation, 'xavier', depth),
        he: simulate(activation, 'he', depth),
      }) as Record<Init, number[]>,
    [depth, activation],
  )

  const [lo, hi] = activation === 'relu' ? [-0.5, 5] : [-2, 2]
  const BINS = 40

  const colors: Record<Init, string> = {
    naive: '#f87171',
    xavier: '#60a5fa',
    he: '#fbbf24',
  }

  return (
    <WidgetFrame
      widgetName="InitDistribution"
      label="activation distribution at layer k"
      right={
        <>
          <span className="font-mono">
            width {WIDTH} · {N_SAMPLES} samples
          </span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            {(['tanh', 'relu'] as Act[]).map((a) => (
              <button
                key={a}
                onClick={() => setActivation(a)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
                  activation === a
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
                )}
              >
                {a}
              </button>
            ))}
          </div>
          <Slider
            label="depth k"
            value={depth}
            min={1}
            max={20}
            step={1}
            onChange={(v) => setDepth(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-dark-accent"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="dead frac"
              value={`${((dists.he.filter((v) => Math.abs(v) < 1e-6).length / dists.he.length) * 100).toFixed(0)}%`}
              accent="text-term-amber"
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 grid grid-cols-3 gap-2 p-2">
        {(['naive', 'xavier', 'he'] as Init[]).map((init) => {
          const hist = histogram(dists[init], BINS, lo, hi)
          const maxH = Math.max(...hist, 1)
          const variance = (() => {
            const arr = dists[init]
            const m = arr.reduce((s, v) => s + v, 0) / arr.length
            let V = 0
            for (const v of arr) V += (v - m) ** 2
            return V / arr.length
          })()
          return (
            <div
              key={init}
              className="border border-dark-border rounded-md overflow-hidden bg-dark-bg flex flex-col"
            >
              <div className="px-3 py-1.5 border-b border-dark-border bg-dark-surface-elevated/40 flex items-center justify-between">
                <span
                  className="text-[11px] font-mono uppercase tracking-wider"
                  style={{ color: colors[init] }}
                >
                  {init === 'naive' ? 'naive (σ=1)' : init + ' init'}
                </span>
                <span className="text-[10px] font-mono text-dark-text-disabled">
                  Var = {variance < 1e-5 || variance > 1e5 ? variance.toExponential(1) : variance.toFixed(3)}
                </span>
              </div>
              <div className="flex-1 flex items-end gap-[1px] px-3 py-3">
                {hist.map((h, i) => (
                  <div
                    key={i}
                    className="flex-1"
                    style={{
                      height: `${(h / maxH) * 100}%`,
                      backgroundColor: colors[init],
                      opacity: 0.7,
                    }}
                  />
                ))}
              </div>
              <div className="flex items-center justify-between px-3 py-1 text-[9px] font-mono text-dark-text-disabled">
                <span>{lo.toFixed(1)}</span>
                <span>0</span>
                <span>{hi.toFixed(1)}</span>
              </div>
            </div>
          )
        })}
      </div>
    </WidgetFrame>
  )
}
