'use client'

import { useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Two horizontal bars driven by E (expert count) and k (top-k). Total
// parameter count grows linearly with E; FLOPs per token only grow with k,
// because the remaining experts are idle for any given token. The Mixtral
// 8x7B preset anchors the abstract numbers to a real model you can look up.

// Per-expert "MLP" parameter count — a stand-in for the large FFN a real MoE
// expert is. We use d_model = 4096, d_ff = 14336 (Mixtral-ish numbers).
const D_MODEL = 4096
const D_FF = 14336
const PER_EXPERT = 2 * D_MODEL * D_FF // up + down projection
const NON_EXPERT = 12 * D_MODEL * D_MODEL // rough estimate of attention + norms per layer
const NUM_LAYERS = 32

function formatBillions(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  return n.toString()
}

const MIXTRAL = { E: 8, k: 2, label: 'Mixtral 8x7B' }

export default function MoEParamvsFlops() {
  const [E, setE] = useState(8)
  const [k, setK] = useState(2)

  // Snap E to allowed ticks
  const eValues = [4, 8, 16, 64]
  const kValues = [1, 2, 4]

  const totalParams = (NON_EXPERT + E * PER_EXPERT) * NUM_LAYERS
  const activeParams = (NON_EXPERT + k * PER_EXPERT) * NUM_LAYERS
  // FLOPs per token ~ 2 × active params (forward pass)
  const flopsPerToken = 2 * activeParams
  const denseFlops = 2 * totalParams // what a dense model with same capacity would spend

  const savings = 1 - flopsPerToken / denseFlops

  const paramBarMax = (NON_EXPERT + 64 * PER_EXPERT) * NUM_LAYERS
  const flopsBarMax = 2 * paramBarMax

  const isMixtral = E === MIXTRAL.E && k === MIXTRAL.k

  return (
    <WidgetFrame
      widgetName="MoEParamvsFlops"
      label="params grow with E · FLOPs grow with k — that's the whole point"
      right={
        <span className="font-mono">
          per-expert ~ 180M · {NUM_LAYERS} layers · d_model={D_MODEL}
        </span>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="E"
            value={E}
            min={4}
            max={64}
            step={1}
            onChange={(v) => setE(eValues.reduce((best, cand) => (Math.abs(cand - v) < Math.abs(best - v) ? cand : best)))}
            format={(v) => `${Math.round(v)} experts`}
            accent="accent-term-cyan"
          />
          <Slider
            label="k"
            value={k}
            min={1}
            max={4}
            step={1}
            onChange={(v) => setK(kValues.reduce((best, cand) => (Math.abs(cand - v) < Math.abs(best - v) ? cand : best)))}
            format={(v) => `top-${Math.round(v)}`}
            accent="accent-term-amber"
          />
          <button
            onClick={() => {
              setE(MIXTRAL.E)
              setK(MIXTRAL.k)
            }}
            className={cn(
              'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
              isMixtral
                ? 'bg-term-amber/20 text-term-amber border border-term-amber'
                : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
            )}
          >
            {MIXTRAL.label}
          </button>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="FLOPs saved" value={`${(savings * 100).toFixed(0)}%`} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 overflow-hidden grid grid-rows-[auto_auto_1fr] gap-4">
        {/* Total parameters bar */}
        <BarRow
          title="total parameters"
          subtitle={`non-expert + ${E} × per-expert`}
          value={totalParams}
          max={paramBarMax}
          fmt={formatBillions}
          color="#67e8f9"
          detail={`${formatBillions(NON_EXPERT * NUM_LAYERS)} base + ${E} × ${formatBillions(PER_EXPERT * NUM_LAYERS)}`}
        />

        {/* FLOPs per token bar */}
        <BarRow
          title="FLOPs per token (forward)"
          subtitle={`non-expert + ${k} × per-expert — the rest are idle`}
          value={flopsPerToken}
          max={flopsBarMax}
          fmt={(n) => `${formatBillions(n)} FLOP`}
          color="#fbbf24"
          detail={`active params ≈ ${formatBillions(activeParams)}`}
        />

        {/* Summary card */}
        <div className="grid grid-cols-3 gap-3 font-mono text-[11px]">
          <SummaryCard
            label="total parameters"
            value={formatBillions(totalParams)}
            hint="what your disk stores"
            accent="text-term-cyan"
          />
          <SummaryCard
            label="active per token"
            value={formatBillions(activeParams)}
            hint={`only ${k} of ${E} experts fire`}
            accent="text-term-amber"
          />
          <SummaryCard
            label="dense equivalent"
            value={formatBillions(2 * totalParams)}
            hint="FLOPs a dense model would spend"
            accent="text-term-rose"
          />
        </div>

        {isMixtral && (
          <div className="absolute bottom-3 left-5 right-5 font-mono text-[10.5px] text-dark-text-muted border-t border-term-amber/30 pt-2">
            <span className="text-term-amber">mixtral 8x7B:</span> 47B total params, but only ~13B
            are active per token — inference cost of a 13B dense, knowledge of a 47B.
          </div>
        )}
      </div>
    </WidgetFrame>
  )
}

function BarRow({
  title,
  subtitle,
  value,
  max,
  fmt,
  color,
  detail,
}: {
  title: string
  subtitle: string
  value: number
  max: number
  fmt: (n: number) => string
  color: string
  detail: string
}) {
  const pct = Math.min(1, value / max)
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between font-mono">
        <div>
          <div className="text-[12px] text-dark-text-primary">{title}</div>
          <div className="text-[10px] text-dark-text-muted">{subtitle}</div>
        </div>
        <div className="text-right">
          <div className="text-[14px] tabular-nums" style={{ color }}>
            {fmt(value)}
          </div>
          <div className="text-[9.5px] text-dark-text-disabled">{detail}</div>
        </div>
      </div>
      <div className="relative h-7 bg-dark-surface-elevated/40 rounded overflow-hidden">
        <div
          className="absolute top-0 bottom-0 rounded transition-all"
          style={{
            width: `${pct * 100}%`,
            backgroundColor: color,
            opacity: 0.7,
          }}
        />
        {/* tick marks */}
        {[0.25, 0.5, 0.75].map((t) => (
          <div
            key={t}
            className="absolute top-0 bottom-0 border-l border-dark-border/60"
            style={{ left: `${t * 100}%` }}
          />
        ))}
      </div>
    </div>
  )
}

function SummaryCard({
  label,
  value,
  hint,
  accent,
}: {
  label: string
  value: string
  hint: string
  accent: string
}) {
  return (
    <div className="bg-dark-surface-elevated/40 rounded p-3 border border-dark-border">
      <div className="text-[9.5px] uppercase tracking-wider text-dark-text-disabled">{label}</div>
      <div className={cn('text-[18px] tabular-nums mt-0.5', accent)}>{value}</div>
      <div className="text-[9.5px] text-dark-text-muted mt-0.5">{hint}</div>
    </div>
  )
}
