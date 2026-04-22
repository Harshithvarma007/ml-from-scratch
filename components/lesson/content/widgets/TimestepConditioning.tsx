'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Sinusoidal timestep embedding visualization. For each t we compute
// embed(t)[2i]   = sin(t / 10000^(2i/D))
// embed(t)[2i+1] = cos(t / 10000^(2i/D))
// with D = 128. The heatmap shows embed across t = 0..T and dims 0..127;
// the slider highlights one row (one t). Below: a cartoon of FiLM
// conditioning — gamma * x + beta applied to a fake 8x8 feature map
// (deterministic 2D noise). Sliders for gamma and beta.

const T_MAX = 1000
const D = 128
const FM = 16

function embed(t: number, d: number): number[] {
  const out: number[] = []
  for (let i = 0; i < d; i++) {
    const idx2 = Math.floor(i / 2)
    const freq = 1 / Math.pow(10000, (2 * idx2) / d)
    out.push(i % 2 === 0 ? Math.sin(t * freq) : Math.cos(t * freq))
  }
  return out
}

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function featureMap(): number[] {
  const rng = mulberry32(7)
  const out: number[] = []
  for (let i = 0; i < FM * FM; i++) {
    out.push((rng() - 0.5) * 1.6)
  }
  return out
}

function embedColor(v: number): string {
  // v in [-1, 1] — amber for +, cyan for -
  const a = Math.min(1, Math.abs(v))
  if (v >= 0) return `rgba(251, 191, 36, ${0.1 + a * 0.85})`
  return `rgba(103, 232, 249, ${0.1 + a * 0.85})`
}

function fmColor(v: number): string {
  const a = Math.min(1, Math.abs(v) / 3)
  if (v >= 0) return `rgba(244, 114, 182, ${0.1 + a * 0.85})`
  return `rgba(167, 139, 250, ${0.1 + a * 0.85})`
}

export default function TimestepConditioning() {
  const [t, setT] = useState(250)
  const [gamma, setGamma] = useState(1.2)
  const [beta, setBeta] = useState(0.2)

  // Heatmap: sample 64 timestep rows across [0, T_MAX] for display.
  const ROW_STEP = 16
  const rows = 64
  const heatmap = useMemo(() => {
    const out: number[][] = []
    for (let r = 0; r < rows; r++) {
      const tr = Math.round((r / (rows - 1)) * T_MAX)
      out.push(embed(tr, D))
    }
    return out
  }, [])
  const selRow = Math.max(0, Math.min(rows - 1, Math.round((t / T_MAX) * (rows - 1))))
  const selEmbed = useMemo(() => embed(t, D), [t])

  // FiLM: gamma * x + beta on the feature map
  const fm = useMemo(featureMap, [])
  const conditioned = fm.map((v) => gamma * v + beta)

  return (
    <WidgetFrame
      widgetName="TimestepConditioning"
      label="timestep embedding + FiLM conditioning — how t talks to the U-Net"
      right={<span className="font-mono">embed(t)_{2}i = sin(t/10000^(2i/D))  ·  FiLM: γ·x + β</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider label="t" value={t} min={0} max={T_MAX} step={1} onChange={(v) => setT(Math.round(v))} format={(v) => String(Math.round(v))} accent="accent-term-pink" />
          <Slider label="γ" value={gamma} min={-2} max={2} step={0.05} onChange={setGamma} format={(v) => v.toFixed(2)} accent="accent-term-amber" />
          <Slider label="β" value={beta} min={-1.5} max={1.5} step={0.05} onChange={setBeta} format={(v) => v.toFixed(2)} accent="accent-term-cyan" />
          <div className="flex items-center gap-3 ml-auto">
            <Readout label="dim 0" value={selEmbed[0].toFixed(3)} accent="text-term-amber" />
            <Readout label="dim 127" value={selEmbed[127].toFixed(3)} accent="text-term-cyan" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-[1.4fr_1fr] gap-4">
        {/* Left: embedding heatmap */}
        <div className="flex flex-col gap-1 min-w-0 min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            embed(t) — rows: t (0 → {T_MAX}) · cols: dim (0 → 127)
          </div>
          <div
            className="relative flex-1 min-h-0 rounded border border-dark-border overflow-hidden"
            style={{
              display: 'grid',
              gridTemplateRows: `repeat(${rows}, 1fr)`,
              gridTemplateColumns: `repeat(${D}, 1fr)`,
            }}
          >
            {heatmap.flatMap((row, r) =>
              row.map((v, c) => (
                <div
                  key={`${r}-${c}`}
                  style={{ backgroundColor: embedColor(v) }}
                  className={r === selRow ? 'outline outline-1 outline-term-pink' : ''}
                />
              )),
            )}
          </div>
          <div className="text-[9.5px] font-mono text-dark-text-muted">
            amber = positive · cyan = negative · selected row outlined
          </div>

          {/* Selected t vector preview */}
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-2">
            embed(t = {t}) — first 32 dims
          </div>
          <div className="flex h-5 gap-[1px] rounded overflow-hidden border border-dark-border">
            {selEmbed.slice(0, 32).map((v, i) => (
              <div key={i} className="flex-1" style={{ backgroundColor: embedColor(v) }} title={`dim ${i} = ${v.toFixed(3)}`} />
            ))}
          </div>
        </div>

        {/* Right: FiLM */}
        <div className="flex flex-col gap-2 min-w-0 min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            FiLM injection — h′ = γ(t)·h + β(t)
          </div>
          <div className="grid grid-cols-2 gap-3 flex-1 min-h-0">
            <FmPanel title="feature h" data={fm} />
            <FmPanel title="conditioned h′" data={conditioned} />
          </div>
          <div className="bg-dark-surface-elevated/40 rounded p-2 font-mono text-[10px] leading-relaxed">
            <div className="text-term-amber">γ = {gamma.toFixed(2)}</div>
            <div className="text-term-cyan">β = {beta.toFixed(2)}</div>
            <div className="text-dark-text-muted mt-1">
              small MLPs turn embed(t) into γ, β per channel — the only way the U-Net
              learns that t=10 and t=900 need different behaviour.
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function FmPanel({ title, data }: { title: string; data: number[] }) {
  return (
    <div className="flex flex-col gap-1 min-w-0 min-h-0">
      <div className="text-[10px] font-mono text-dark-text-muted text-center">{title}</div>
      <div
        className="flex-1 min-h-0 aspect-square w-full max-w-[180px] self-center rounded border border-dark-border overflow-hidden"
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${FM}, 1fr)`,
          gridTemplateRows: `repeat(${FM}, 1fr)`,
        }}
      >
        {data.map((v, i) => (
          <div key={i} style={{ backgroundColor: fmColor(v) }} />
        ))}
      </div>
    </div>
  )
}
