'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Show a (batch × features) matrix. Toggle "raw" / "LayerNorm" and watch
// each row (one example) independently rescale to mean 0, variance 1. Click
// a row to highlight — the bottom bar shows its mean, std, and the per-element
// normalisation calculation.

const BATCH = 6
const FEATURES = 8

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gauss(rng: () => number, std: number, mean: number): number {
  const u = Math.max(rng(), 1e-9)
  const v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * std + mean
}

function makeRaw(scale: number): number[][] {
  const rng = mulberry32(11)
  const out: number[][] = []
  for (let i = 0; i < BATCH; i++) {
    // Each example gets a different mean + std — this is the reality that
    // LayerNorm is designed to fix.
    const mean = (i - BATCH / 2) * scale
    const std = 0.5 + (i * 0.5)
    const row: number[] = []
    for (let j = 0; j < FEATURES; j++) row.push(gauss(rng, std, mean))
    out.push(row)
  }
  return out
}

function layerNorm(data: number[][], eps = 1e-5): number[][] {
  return data.map((row) => {
    const n = row.length
    const mean = row.reduce((s, v) => s + v, 0) / n
    const variance = row.reduce((s, v) => s + (v - mean) ** 2, 0) / n
    const std = Math.sqrt(variance + eps)
    return row.map((v) => (v - mean) / std)
  })
}

export default function LayerNormVisualization() {
  const [scale, setScale] = useState(1.2)
  const [normalized, setNormalized] = useState(false)
  const [selectedRow, setSelectedRow] = useState(2)

  const raw = useMemo(() => makeRaw(scale), [scale])
  const normed = useMemo(() => layerNorm(raw), [raw])
  const data = normalized ? normed : raw

  const rowStats = useMemo(() => {
    return data.map((row) => {
      const mean = row.reduce((s, v) => s + v, 0) / row.length
      const variance = row.reduce((s, v) => s + (v - mean) ** 2, 0) / row.length
      return { mean, std: Math.sqrt(variance) }
    })
  }, [data])

  const absMax = Math.max(...data.flat().map(Math.abs), 1)

  return (
    <WidgetFrame
      widgetName="LayerNormVisualization"
      label="layer normalization — each example normalized independently"
      right={
        <>
          <span className="font-mono">
            shape: ({BATCH}, {FEATURES})
          </span>
          <span className="text-dark-text-disabled">·</span>
          <span className="font-mono">axis=-1</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            <button
              onClick={() => setNormalized(false)}
              className={cn(
                'px-2.5 py-1 rounded text-[11px] font-mono uppercase transition-all',
                !normalized
                  ? 'bg-dark-accent text-white'
                  : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
              )}
            >
              raw
            </button>
            <button
              onClick={() => setNormalized(true)}
              className={cn(
                'px-2.5 py-1 rounded text-[11px] font-mono uppercase transition-all',
                normalized
                  ? 'bg-dark-accent text-white'
                  : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
              )}
            >
              LayerNorm
            </button>
          </div>
          <Slider
            label="input spread"
            value={scale}
            min={0.2}
            max={3}
            step={0.05}
            onChange={setScale}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="row mean"
              value={rowStats[selectedRow].mean.toFixed(3)}
              accent={normalized ? 'text-term-green' : 'text-term-rose'}
            />
            <Readout
              label="row std"
              value={rowStats[selectedRow].std.toFixed(3)}
              accent={normalized ? 'text-term-green' : 'text-term-rose'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 overflow-auto">
        <div className="max-w-[900px] mx-auto">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
            batch (rows) × features (cols) · click a row to inspect
          </div>
          <div className="flex items-stretch gap-3">
            {/* Feature index header + data matrix */}
            <div className="flex-1 space-y-1">
              <div className="grid grid-cols-1 md:grid-cols-[80px_1fr] gap-3 items-center">
                <div />
                <div
                  className="grid gap-1 text-[9px] text-dark-text-disabled text-center font-mono"
                  style={{ gridTemplateColumns: `repeat(${FEATURES}, 1fr)` }}
                >
                  {Array.from({ length: FEATURES }).map((_, j) => (
                    <span key={j}>f{sub(j)}</span>
                  ))}
                </div>
              </div>
              {data.map((row, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedRow(i)}
                  className={cn(
                    'w-full grid grid-cols-1 md:grid-cols-[80px_1fr] gap-3 items-center px-1 py-1 rounded transition-colors',
                    selectedRow === i ? 'bg-dark-accent/15 ring-1 ring-dark-accent/40' : 'hover:bg-white/[0.02]'
                  )}
                >
                  <div className="flex items-center justify-between font-mono text-[11px]">
                    <span className="text-dark-text-muted">ex{sub(i)}</span>
                    <span
                      className={cn(
                        'text-[10px] tabular-nums',
                        normalized ? 'text-term-green' : 'text-dark-text-disabled'
                      )}
                    >
                      μ={rowStats[i].mean.toFixed(2)}
                    </span>
                  </div>
                  <div
                    className="grid gap-1"
                    style={{ gridTemplateColumns: `repeat(${FEATURES}, 1fr)` }}
                  >
                    {row.map((v, j) => (
                      <div
                        key={j}
                        className="relative h-6 bg-dark-bg rounded border border-dark-border overflow-hidden"
                      >
                        <div
                          className={cn(
                            'absolute inset-y-0',
                            v >= 0 ? 'left-1/2 bg-term-amber/60' : 'right-1/2 bg-term-rose/50'
                          )}
                          style={{
                            width: `${((Math.abs(v) / absMax) * 50).toFixed(3)}%`,
                          }}
                        />
                        <span className="absolute inset-0 flex items-center justify-center text-[9px] tabular-nums text-dark-text-primary font-mono">
                          {v.toFixed(1)}
                        </span>
                      </div>
                    ))}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Formula + per-element calc for the selected row */}
          <div className="mt-5 p-4 rounded border border-dark-border bg-dark-surface-elevated/30 font-mono text-[11.5px]">
            <div className="text-[10px] uppercase tracking-wider text-dark-text-disabled mb-2">
              the operation applied to row {selectedRow}
            </div>
            <div className="mb-2 text-dark-text-primary">
              μ = (1/D) · Σⱼ xⱼ = {rowStats[selectedRow].mean.toFixed(3)} ·{' '}
              σ² = (1/D) · Σⱼ (xⱼ − μ)² = {(rowStats[selectedRow].std ** 2).toFixed(3)} ·{' '}
              x&apos;ⱼ = (xⱼ − μ) / √(σ² + ε)
            </div>
            <div className="text-[10.5px] text-dark-text-muted leading-relaxed">
              The row&apos;s mean is subtracted and it&apos;s rescaled by its std. After
              LayerNorm, every example has mean ≈ 0 and std ≈ 1 —{' '}
              <span className="text-term-green">independently of the batch</span>. Compare a
              batch-of-1 inference call vs a batch-of-1000 training call: identical
              normalization either way.
            </div>
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
