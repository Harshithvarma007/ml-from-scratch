'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Weight histogram with a few extreme outliers in a couple of channels.
// Four quantization strategies let the user see how MSE changes:
//   - naive per-tensor  (one scale for the whole matrix → small weights collapse)
//   - per-channel       (one scale per column → outliers still blow up that col)
//   - mixed-precision   (outlier channels kept in fp16, rest in int8)
//   - SmoothQuant       (channel-wise pre-scaling to smooth the activations)
//
// We fabricate a tiny d × C weight tensor where 2 columns contain outliers,
// compute int8 round-trip under each strategy, and show the resulting
// histogram along with a per-strategy MSE bar.

const D = 200 // rows per channel
const C = 12  // channels (columns)
const OUTLIER_CHANNELS = [3, 8]

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

const W: number[][] = (() => {
  const rng = mulberry32(21)
  const out: number[][] = []
  for (let r = 0; r < D; r++) {
    const row: number[] = []
    for (let c = 0; c < C; c++) {
      const base = gauss(rng) * 0.2
      if (OUTLIER_CHANNELS.includes(c) && rng() < 0.04) {
        row.push(base + (rng() > 0.5 ? 4.0 : -4.0))
      } else {
        row.push(base)
      }
    }
    out.push(row)
  }
  return out
})()

function quantInt8(val: number, scale: number): number {
  const code = Math.max(-127, Math.min(127, Math.round(val / scale)))
  return code * scale
}

function perTensor(): number[][] {
  let m = 0
  for (const row of W) for (const v of row) m = Math.max(m, Math.abs(v))
  const scale = m / 127
  return W.map((row) => row.map((v) => quantInt8(v, scale)))
}

function perChannel(): number[][] {
  const scales = new Array(C).fill(0)
  for (let c = 0; c < C; c++) {
    let m = 0
    for (let r = 0; r < D; r++) m = Math.max(m, Math.abs(W[r][c]))
    scales[c] = m / 127
  }
  return W.map((row) => row.map((v, c) => quantInt8(v, scales[c] || 1)))
}

function mixedPrecision(): number[][] {
  // Non-outlier channels quantized per-channel. Outlier channels kept in fp16
  // (modeled as no round-trip error).
  const scales = new Array(C).fill(0)
  for (let c = 0; c < C; c++) {
    if (OUTLIER_CHANNELS.includes(c)) continue
    let m = 0
    for (let r = 0; r < D; r++) m = Math.max(m, Math.abs(W[r][c]))
    scales[c] = m / 127
  }
  return W.map((row) =>
    row.map((v, c) => (OUTLIER_CHANNELS.includes(c) ? v : quantInt8(v, scales[c] || 1))),
  )
}

function smoothQuant(): number[][] {
  // Scale each channel down by its max so all channels share a similar
  // magnitude, quantize, then inverse-scale. Equivalent to a single scalar
  // rebalance per channel that offloads the outliers into the activations
  // side of the matmul. MSE should be closer to per-channel but with less
  // collateral damage to small-magnitude channels.
  const scales = new Array(C).fill(0)
  for (let c = 0; c < C; c++) {
    let m = 0
    for (let r = 0; r < D; r++) m = Math.max(m, Math.abs(W[r][c]))
    scales[c] = Math.pow(Math.max(m, 0.01), 0.5) // sqrt smoothing factor
  }
  // Effective tensor: W / s; quantize; restore
  const pre: number[][] = W.map((row) => row.map((v, c) => v / scales[c]))
  // quantize per-channel (all channels now comparable in magnitude)
  const preScales = new Array(C).fill(0)
  for (let c = 0; c < C; c++) {
    let m = 0
    for (let r = 0; r < D; r++) m = Math.max(m, Math.abs(pre[r][c]))
    preScales[c] = m / 127
  }
  return pre.map((row) => row.map((v, c) => quantInt8(v, preScales[c] || 1) * scales[c]))
}

const STRATEGIES = [
  { key: 'per-tensor', name: 'naive per-tensor', run: perTensor, color: '#fb7185' },
  { key: 'per-channel', name: 'per-channel', run: perChannel, color: '#fbbf24' },
  { key: 'mixed', name: 'mixed-precision', run: mixedPrecision, color: '#a78bfa' },
  { key: 'smoothquant', name: 'SmoothQuant', run: smoothQuant, color: '#4ade80' },
] as const

type StrategyKey = typeof STRATEGIES[number]['key']

function computeMSE(q: number[][]): number {
  let s = 0
  let n = 0
  for (let r = 0; r < D; r++) {
    for (let c = 0; c < C; c++) {
      const d = W[r][c] - q[r][c]
      s += d * d
      n += 1
    }
  }
  return s / n
}

function histogram(flat: number[], lo: number, hi: number, bins: number): number[] {
  const out = new Array(bins).fill(0)
  const step = (hi - lo) / bins
  for (const v of flat) {
    if (v < lo || v > hi) continue
    const idx = Math.min(bins - 1, Math.floor((v - lo) / step))
    out[idx] += 1
  }
  return out
}

export default function OutlierHandling() {
  const [strat, setStrat] = useState<StrategyKey>('per-tensor')

  const results = useMemo(() => {
    return STRATEGIES.reduce<Record<StrategyKey, { q: number[][]; mse: number }>>(
      (acc, s) => {
        const q = s.run()
        acc[s.key] = { q, mse: computeMSE(q) }
        return acc
      },
      {} as Record<StrategyKey, { q: number[][]; mse: number }>,
    )
  }, [])

  const current = results[strat]
  const hi = 5
  const lo = -5
  const BINS = 60
  const origFlat = W.flat()
  const qFlat = current.q.flat()
  const origHist = histogram(origFlat, lo, hi, BINS)
  const qHist = histogram(qFlat, lo, hi, BINS)
  const maxBin = Math.max(...origHist, ...qHist, 1)

  const maxMse = Math.max(...STRATEGIES.map((s) => results[s.key].mse), 1e-9)

  return (
    <WidgetFrame
      widgetName="OutlierHandling"
      label="outliers and quantization — toggle strategy, watch the MSE collapse"
      right={<span className="font-mono">d = {D}·{C} · outlier channels: {OUTLIER_CHANNELS.join(', ')}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5 flex-wrap">
            {STRATEGIES.map((s) => (
              <button
                key={s.key}
                onClick={() => setStrat(s.key)}
                className={cn(
                  'px-2.5 py-1 rounded text-[10.5px] font-mono transition-all border',
                  strat === s.key
                    ? 'border-dark-accent text-white bg-dark-accent'
                    : 'border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {s.name}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="mse"
              value={current.mse.toExponential(2)}
              accent={strat === 'per-tensor' ? 'text-term-rose' : 'text-term-green'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <div className="grid h-full grid-cols-1 md:grid-cols-[1fr_260px] gap-4 min-h-0">
          {/* Left: histograms */}
          <div className="flex flex-col gap-2 min-h-0">
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
              original weights (cyan) · dequantized (amber) — {STRATEGIES.find((s) => s.key === strat)?.name}
            </div>
            <div className="flex-1 min-h-0 rounded border border-dark-border/60 bg-dark-bg/60 overflow-hidden">
              <svg viewBox="0 0 600 260" preserveAspectRatio="none" className="w-full h-full">
                <Histogram bins={origHist} maxBin={maxBin} color="#67e8f9" offsetY={10} />
                <Histogram bins={qHist} maxBin={maxBin} color="#fbbf24" offsetY={140} />
                <text x={10} y={22} fontSize="9" fill="#67e8f9" fontFamily="JetBrains Mono, monospace">
                  original (fp32)
                </text>
                <text x={10} y={152} fontSize="9" fill="#fbbf24" fontFamily="JetBrains Mono, monospace">
                  int8 round-trip
                </text>
                {/* Axis */}
                {[-4, -2, 0, 2, 4].map((x) => {
                  const sx = 30 + ((x - lo) / (hi - lo)) * 560
                  return (
                    <g key={x}>
                      <line x1={sx} y1={128} x2={sx} y2={135} stroke="#333" strokeWidth={1} />
                      <text x={sx} y={256} fontSize="9" textAnchor="middle" fill="#555" fontFamily="JetBrains Mono, monospace">
                        {x}
                      </text>
                    </g>
                  )
                })}
              </svg>
            </div>

            {/* Channel-level view (condensed) */}
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-1">
              channel magnitudes (max |w| per column)
            </div>
            <div className="grid gap-1 h-12" style={{ gridTemplateColumns: `repeat(${C}, 1fr)` }}>
              {Array.from({ length: C }).map((_, c) => {
                let m = 0
                for (let r = 0; r < D; r++) m = Math.max(m, Math.abs(W[r][c]))
                const isOutlier = OUTLIER_CHANNELS.includes(c)
                return (
                  <div key={c} className="relative flex items-end" title={`ch ${c} · max = ${m.toFixed(2)}`}>
                    <div
                      className={cn('w-full rounded-sm', isOutlier ? 'bg-term-rose/80' : 'bg-term-cyan/70')}
                      style={{ height: `${Math.min(100, (m / 5) * 100)}%` }}
                    />
                  </div>
                )
              })}
            </div>
          </div>

          {/* Right: MSE bars + explainer */}
          <div className="flex flex-col gap-2 min-w-0">
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
              mse by strategy
            </div>
            <div className="flex flex-col gap-1.5 rounded border border-dark-border bg-dark-surface-elevated/40 p-3">
              {STRATEGIES.map((s) => {
                const m = results[s.key].mse
                const pct = (m / maxMse) * 100
                const active = strat === s.key
                return (
                  <div
                    key={s.key}
                    className={cn(
                      'flex flex-col gap-0.5 font-mono text-[10.5px]',
                      active && 'text-term-amber',
                    )}
                  >
                    <div className="flex justify-between">
                      <span>{s.name}</span>
                      <span className="tabular-nums">{m.toExponential(2)}</span>
                    </div>
                    <div className="h-2 bg-dark-surface-elevated/60 rounded-sm overflow-hidden">
                      <div
                        className="h-full rounded-sm"
                        style={{
                          width: `${pct}%`,
                          backgroundColor: s.color,
                          opacity: active ? 1 : 0.65,
                        }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>

            <div className="rounded border border-dark-border bg-dark-surface-elevated/40 p-3 font-mono text-[10.5px] leading-relaxed text-dark-text-muted">
              {strat === 'per-tensor' && (
                <>
                  one <span className="text-term-rose">scale</span> for every weight. A single 4.0 outlier forces scale ≈ 4/127,
                  which quantizes the 99% of small weights to just a handful of codes. MSE is dominated by lost precision.
                </>
              )}
              {strat === 'per-channel' && (
                <>
                  a separate <span className="text-term-amber">scale per column</span>. The outlier column still gets a coarse scale,
                  but the quiet channels recover their full int8 resolution.
                </>
              )}
              {strat === 'mixed' && (
                <>
                  keep the <span className="text-term-purple">outlier columns in fp16</span>. Everything else runs in int8.
                  Costs a few % memory but the MSE contribution from outliers vanishes.
                </>
              )}
              {strat === 'smoothquant' && (
                <>
                  pre-scale the weights by <span className="text-term-green">sqrt(max|w|)</span> per channel and absorb the
                  inverse into activations. The quantizer sees a flatter distribution.
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function Histogram({
  bins,
  maxBin,
  color,
  offsetY,
}: {
  bins: number[]
  maxBin: number
  color: string
  offsetY: number
}) {
  const padL = 30
  const padR = 10
  const plotW = 600 - padL - padR
  const plotH = 110
  const binW = plotW / bins.length
  return (
    <g>
      {bins.map((c, i) => {
        const h = (c / maxBin) * plotH
        return (
          <rect
            key={i}
            x={padL + i * binW}
            y={offsetY + plotH - h}
            width={Math.max(1, binW - 1)}
            height={h}
            fill={color}
            opacity={0.8}
          />
        )
      })}
      <line
        x1={padL}
        y1={offsetY + plotH}
        x2={padL + plotW}
        y2={offsetY + plotH}
        stroke="#2a2a32"
        strokeWidth={1}
      />
    </g>
  )
}
