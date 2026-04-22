'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Side-by-side: same input vector. Left panel runs LayerNorm. Right panel
// runs RMSNorm. Both show the intermediate quantities (mean, var, rms) and
// the output. A slider shifts the whole input uniformly, demonstrating: RMS
// ignores a constant offset; LayerNorm cancels it.

const D = 8

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

function makeBase(seed = 4): number[] {
  const rng = mulberry32(seed)
  return Array.from({ length: D }, () => gauss(rng) * 0.9)
}

export default function RMSvsLayerNorm() {
  const [offset, setOffset] = useState(0)
  const base = useMemo(() => makeBase(), [])

  const x = useMemo(() => base.map((v) => v + offset), [base, offset])

  const layerNorm = useMemo(() => {
    const mean = x.reduce((s, v) => s + v, 0) / D
    const variance = x.reduce((s, v) => s + (v - mean) ** 2, 0) / D
    const std = Math.sqrt(variance + 1e-5)
    return {
      mean,
      variance,
      std,
      output: x.map((v) => (v - mean) / std),
    }
  }, [x])

  const rmsNorm = useMemo(() => {
    const ms = x.reduce((s, v) => s + v * v, 0) / D
    const rms = Math.sqrt(ms + 1e-5)
    return {
      rms,
      output: x.map((v) => v / rms),
    }
  }, [x])

  const absMax = Math.max(
    ...x.map(Math.abs),
    ...layerNorm.output.map(Math.abs),
    ...rmsNorm.output.map(Math.abs),
    1,
  )

  return (
    <WidgetFrame
      widgetName="RMSvsLayerNorm"
      label="LayerNorm vs RMSNorm — the mean subtraction is the only difference"
      right={
        <>
          <span className="font-mono">same input · both outputs shown</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="bulk offset"
            value={offset}
            min={-3}
            max={3}
            step={0.05}
            onChange={setOffset}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="input μ" value={(x.reduce((s, v) => s + v, 0) / D).toFixed(3)} />
            <Readout label="input rms" value={rmsNorm.rms.toFixed(3)} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 overflow-auto">
        <div className="max-w-[920px] mx-auto space-y-5">
          {/* Input row */}
          <Row label="input x" values={x} color="#67e8f9" absMax={absMax} />

          <div className="grid grid-cols-2 gap-4">
            {/* LayerNorm */}
            <div className="rounded border border-dark-border bg-dark-surface-elevated/30 p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-mono uppercase tracking-wider text-term-purple">
                  LayerNorm
                </span>
                <span className="text-[10px] font-mono text-dark-text-disabled">
                  (x − μ) / √(σ² + ε)
                </span>
              </div>
              <StatLine label="μ = mean" value={layerNorm.mean.toFixed(4)} />
              <StatLine label="σ² = var" value={layerNorm.variance.toFixed(4)} />
              <StatLine label="√(σ² + ε)" value={layerNorm.std.toFixed(4)} />
              <div className="mt-3">
                <Row
                  label="output"
                  values={layerNorm.output}
                  color="#a78bfa"
                  absMax={absMax}
                  compact
                />
              </div>
            </div>

            {/* RMSNorm */}
            <div className="rounded border border-dark-border bg-dark-surface-elevated/30 p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-mono uppercase tracking-wider text-term-amber">
                  RMSNorm
                </span>
                <span className="text-[10px] font-mono text-dark-text-disabled">
                  x / √(mean(x²) + ε)
                </span>
              </div>
              <StatLine label="μ = mean" value="(not used)" muted />
              <StatLine label="mean(x²)" value={(rmsNorm.rms ** 2).toFixed(4)} />
              <StatLine label="rms" value={rmsNorm.rms.toFixed(4)} />
              <div className="mt-3">
                <Row label="output" values={rmsNorm.output} color="#fbbf24" absMax={absMax} compact />
              </div>
            </div>
          </div>

          <div className="p-3 rounded border border-dark-border bg-dark-bg text-[11.5px] font-sans text-dark-text-secondary leading-relaxed">
            <strong className="text-dark-text-primary">Drag the offset slider.</strong>{' '}
            LayerNorm&apos;s output is unchanged — the mean subtraction kills any constant
            bulk shift. RMSNorm&apos;s output <em>does</em> shift with the offset. In
            practice this matters less than you&apos;d think: the immediately-following
            learned <code>γ</code> absorbs most of it, and empirically RMSNorm trains just
            as well while skipping the mean computation entirely.
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function StatLine({ label, value, muted }: { label: string; value: string; muted?: boolean }) {
  return (
    <div className="flex items-center justify-between font-mono text-[11px] py-0.5">
      <span className={cn('text-dark-text-muted', muted && 'line-through opacity-50')}>
        {label}
      </span>
      <span className={cn('tabular-nums', muted ? 'text-dark-text-disabled line-through' : 'text-dark-text-primary')}>
        {value}
      </span>
    </div>
  )
}

function Row({
  label,
  values,
  color,
  absMax,
  compact,
}: {
  label: string
  values: number[]
  color: string
  absMax: number
  compact?: boolean
}) {
  return (
    <div className={cn('flex items-center gap-3', compact ? 'text-[10px]' : 'text-[11px]')}>
      <span className="text-dark-text-muted font-mono w-14">{label}</span>
      <div
        className="flex-1 grid gap-1"
        style={{ gridTemplateColumns: `repeat(${values.length}, 1fr)` }}
      >
        {values.map((v, i) => (
          <div key={i} className="relative h-6 bg-dark-bg rounded border border-dark-border overflow-hidden">
            <div
              className="absolute inset-y-0 rounded"
              style={{
                backgroundColor: color + '99',
                left: v >= 0 ? '50%' : `${(50 + (v / absMax) * 50).toFixed(3)}%`,
                width: `${((Math.abs(v) / absMax) * 50).toFixed(3)}%`,
              }}
            />
            <span className="absolute inset-0 flex items-center justify-center text-[9px] tabular-nums text-dark-text-primary font-mono">
              {v.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
