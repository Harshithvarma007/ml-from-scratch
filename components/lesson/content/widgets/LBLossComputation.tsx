'use client'

import { useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// A calculator for the Switch-Transformer-style auxiliary loss:
//
//   L_aux = α · N · Σ_i f_i · P_i
//
// where N is the number of experts, f_i is the fraction of tokens that were
// routed (top-1) to expert i, and P_i is the mean router probability for
// expert i. Presets flip between balanced, mild-skew, and collapsed regimes.
// The per-expert f·P contribution is a bar — you can see who's dragging the
// loss upward.

const NUM_EXPERTS = 8

type Preset = { name: string; f: number[]; P: number[]; blurb: string }

const PRESETS: Preset[] = [
  {
    name: 'balanced',
    f: new Array(NUM_EXPERTS).fill(1 / NUM_EXPERTS),
    P: new Array(NUM_EXPERTS).fill(1 / NUM_EXPERTS),
    blurb: 'every expert sees 12.5% of tokens · router bets uniformly',
  },
  {
    name: 'mild-skew',
    f: [0.22, 0.18, 0.14, 0.11, 0.1, 0.1, 0.08, 0.07],
    P: [0.2, 0.17, 0.15, 0.12, 0.11, 0.1, 0.08, 0.07],
    blurb: 'natural variance — some experts favored, most still in play',
  },
  {
    name: 'collapsed',
    f: [0.52, 0.28, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01],
    P: [0.48, 0.27, 0.1, 0.06, 0.04, 0.03, 0.01, 0.01],
    blurb: 'two experts see 80% of tokens · LB loss better be large',
  },
]

function lbLoss(f: number[], P: number[], alpha: number): { total: number; per: number[] } {
  const N = f.length
  const per = f.map((fi, i) => fi * P[i])
  const sum = per.reduce((a, b) => a + b, 0)
  return { total: alpha * N * sum, per }
}

export default function LBLossComputation() {
  const [preset, setPreset] = useState<Preset>(PRESETS[1])
  const [alpha, setAlpha] = useState(0.01)

  const { total, per } = lbLoss(preset.f, preset.P, alpha)
  const uniform = 1 / NUM_EXPERTS
  const fUniform = lbLoss(
    new Array(NUM_EXPERTS).fill(uniform),
    new Array(NUM_EXPERTS).fill(uniform),
    alpha,
  ).total

  return (
    <WidgetFrame
      widgetName="LBLossComputation"
      label="L_aux = α · N · Σᵢ fᵢ · Pᵢ — auxiliary load-balancing loss"
      right={
        <span className="font-mono">
          Switch Transformer · N = {NUM_EXPERTS} experts
        </span>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="α"
            value={alpha}
            min={0}
            max={0.1}
            step={0.002}
            onChange={setAlpha}
            format={(v) => v.toFixed(3)}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-1.5">
            {PRESETS.map((p) => (
              <button
                key={p.name}
                onClick={() => setPreset(p)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
                  preset.name === p.name
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {p.name}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="Σ fᵢ·Pᵢ" value={per.reduce((a, b) => a + b, 0).toFixed(4)} accent="text-term-cyan" />
            <Readout label="L_aux" value={total.toExponential(2)} accent="text-term-pink" />
            <Readout
              label="vs uniform"
              value={`${(total / (fUniform || 1e-9)).toFixed(1)}×`}
              accent={total > fUniform * 1.5 ? 'text-term-rose' : 'text-term-green'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-1 md:grid-cols-[1fr_260px] gap-4">
        {/* Left: per-expert contribution bars + two rows for f, P */}
        <div className="flex flex-col gap-2 min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            {preset.blurb}
          </div>
          <Row
            label="fᵢ"
            sublabel="token fraction routed to expert i"
            values={preset.f}
            max={0.6}
            color="#67e8f9"
            fmt={(v) => (v * 100).toFixed(1) + '%'}
          />
          <Row
            label="Pᵢ"
            sublabel="mean router probability for expert i"
            values={preset.P}
            max={0.6}
            color="#fbbf24"
            fmt={(v) => v.toFixed(3)}
          />
          <Row
            label="fᵢ · Pᵢ"
            sublabel="per-expert loss contribution — big bar = culprit"
            values={per}
            max={Math.max(0.1, Math.max(...per) * 1.2)}
            color="#f472b6"
            fmt={(v) => v.toFixed(4)}
            highlight
          />
        </div>

        {/* Right: formula breakdown + computed values */}
        <div className="flex flex-col gap-3 min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            formula
          </div>
          <div className="bg-dark-surface-elevated/40 rounded p-3 font-mono text-[11px] leading-relaxed space-y-2">
            <div>
              <span className="text-term-pink">L_aux</span>
              <span className="text-dark-text-muted"> = </span>
              <span className="text-term-amber">α</span>
              <span className="text-dark-text-muted"> · </span>
              <span className="text-dark-text-primary">N</span>
              <span className="text-dark-text-muted"> · </span>
              <span className="text-term-cyan">Σᵢ fᵢ · Pᵢ</span>
            </div>
            <div className="text-[10px] text-dark-text-muted">
              minimized when f and P are both uniform → L_aux = α
            </div>
            <div className="border-t border-dark-border pt-2 space-y-1 text-[10.5px]">
              <Eq label="α" value={alpha.toFixed(4)} color="#fbbf24" />
              <Eq label="N" value={String(NUM_EXPERTS)} color="#e5e7eb" />
              <Eq label="Σ fᵢ·Pᵢ" value={per.reduce((a, b) => a + b, 0).toFixed(5)} color="#67e8f9" />
              <div className="border-t border-dark-border my-1" />
              <Eq label="L_aux" value={total.toExponential(3)} color="#f472b6" bold />
            </div>
          </div>

          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-1">
            reference
          </div>
          <div className="bg-dark-surface-elevated/40 rounded p-3 font-mono text-[10px] leading-relaxed text-dark-text-muted">
            <div>
              uniform baseline (f = P = 1/N):&nbsp;
              <span className="text-term-green">{fUniform.toExponential(2)}</span>
            </div>
            <div>
              current / uniform:&nbsp;
              <span className={total > fUniform * 1.5 ? 'text-term-rose' : 'text-term-green'}>
                {(total / (fUniform || 1e-9)).toFixed(2)}×
              </span>
            </div>
            <div className="mt-1 text-dark-text-disabled">
              the gradient ∂L_aux/∂router pushes each expert toward f = P = 1/N.
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function Row({
  label,
  sublabel,
  values,
  max,
  color,
  fmt,
  highlight,
}: {
  label: string
  sublabel: string
  values: number[]
  max: number
  color: string
  fmt: (v: number) => string
  highlight?: boolean
}) {
  return (
    <div
      className={cn(
        'flex flex-col gap-1 min-h-0',
        highlight && 'bg-dark-surface-elevated/40 rounded px-2 py-1.5 -mx-2',
      )}
    >
      <div className="flex items-baseline justify-between font-mono">
        <div>
          <span className="text-[12px]" style={{ color }}>{label}</span>
          <span className="text-[9.5px] text-dark-text-muted ml-2">{sublabel}</span>
        </div>
      </div>
      <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${NUM_EXPERTS}, 1fr)` }}>
        {values.map((v, i) => {
          const pct = Math.min(1, v / max)
          return (
            <div key={i} className="flex flex-col gap-0.5 min-w-0">
              <div className="relative h-5 bg-dark-bg rounded-sm overflow-hidden">
                <div
                  className="absolute left-0 bottom-0 top-0 transition-all"
                  style={{ width: `${pct * 100}%`, backgroundColor: color, opacity: 0.85 }}
                />
              </div>
              <div className="flex items-center justify-between font-mono text-[9px]">
                <span className="text-dark-text-disabled">E{i}</span>
                <span className="tabular-nums" style={{ color }}>
                  {fmt(v)}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function Eq({ label, value, color, bold }: { label: string; value: string; color: string; bold?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-dark-text-secondary">{label}</span>
      <span className={cn('tabular-nums', bold && 'font-bold')} style={{ color }}>
        {value}
      </span>
    </div>
  )
}
