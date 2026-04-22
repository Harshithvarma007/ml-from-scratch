'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Three 32x32 tiles side-by-side: the noisy input x_t, the model's predicted
// noise ε_hat, and the residual x_t - sqrt(1-alpha_bar) * eps_hat (a rough
// reconstruction). A single slider controls "model quality" q in [0, 1]:
//   q = 0  → model predicts all zeros (worst case)
//   q = 1  → model perfectly recovers the true noise
// Intermediate q linearly interpolates between the two. Shows per-pixel MSE.

const N = 32
const BETA_T = 0.55
const ALPHA_BAR = 1 - BETA_T

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

function cleanImage(): number[] {
  const out = new Array(N * N).fill(-0.5)
  // concentric rings + a diagonal bar — keeps the scene recognizable.
  const cx = 15
  const cy = 15
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const d = Math.hypot(x - cx, y - cy)
      let v = -0.5
      if (d < 4) v = 0.9
      else if (d < 7) v = -0.3
      else if (d < 10) v = 0.5
      else if (d < 13) v = -0.1
      if (Math.abs(x - y) < 2 && x > 18) v = 0.7
      out[y * N + x] = v
    }
  }
  return out
}

function toGray(v: number): string {
  const g = Math.max(0, Math.min(255, Math.round((v * 0.5 + 0.5) * 255)))
  return `rgb(${g},${g},${g})`
}

function diffColor(v: number): string {
  // Residual/error viz: positive → cyan, negative → rose, magnitude → intensity.
  const a = Math.min(1, Math.abs(v) * 2.5)
  if (v >= 0) return `rgba(103, 232, 249, ${a})`
  return `rgba(251, 113, 133, ${a})`
}

export default function DenoisingTaskSim() {
  const [q, setQ] = useState(0.6)

  const { x_t, eps_true, eps_hat, residual, mse } = useMemo(() => {
    const x0 = cleanImage()
    const rng = mulberry32(11)
    const eps_true = new Array(N * N)
    for (let i = 0; i < N * N; i++) eps_true[i] = gauss(rng)
    const x_t = new Array(N * N)
    for (let i = 0; i < N * N; i++) {
      x_t[i] = Math.sqrt(ALPHA_BAR) * x0[i] + Math.sqrt(1 - ALPHA_BAR) * eps_true[i]
    }
    // Model quality: q blends zero predictor and the oracle predictor.
    const eps_hat = new Array(N * N)
    for (let i = 0; i < N * N; i++) eps_hat[i] = q * eps_true[i]
    // Residual used to recover x_0 from x_t:   x_0 ≈ (x_t - sqrt(1-ab)*eps_hat) / sqrt(ab)
    const residual = new Array(N * N)
    let err = 0
    for (let i = 0; i < N * N; i++) {
      residual[i] = (x_t[i] - Math.sqrt(1 - ALPHA_BAR) * eps_hat[i]) / Math.sqrt(ALPHA_BAR)
      const d = eps_hat[i] - eps_true[i]
      err += d * d
    }
    return { x_t, eps_true, eps_hat, residual, mse: err / (N * N) }
  }, [q])

  return (
    <WidgetFrame
      widgetName="DenoisingTaskSim"
      label="the denoising task — input, predicted noise, reconstruction"
      right={<span className="font-mono">loss = ‖ε̂ − ε‖² · β = {BETA_T.toFixed(2)}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="model quality q"
            value={q}
            min={0}
            max={1}
            step={0.01}
            onChange={setQ}
            format={(v) => v.toFixed(2)}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="mse(ε̂, ε)"
              value={mse.toFixed(3)}
              accent={mse < 0.2 ? 'text-term-green' : mse > 0.7 ? 'text-term-rose' : 'text-term-amber'}
            />
            <Readout
              label="recon quality"
              value={q < 0.3 ? 'bad' : q < 0.7 ? 'okay' : 'near-oracle'}
              accent={q < 0.3 ? 'text-term-rose' : q < 0.7 ? 'text-term-amber' : 'text-term-green'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden flex flex-col gap-3">
        <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
          q = 0 predicts zeros (worst) · q = 1 recovers the exact noise (oracle)
        </div>
        <div className="flex-1 min-h-0 flex items-center justify-around gap-4">
          <TilePanel
            title="noisy input x_t"
            sub={`sqrt(ᾱ)·x₀ + sqrt(1−ᾱ)·ε`}
            data={x_t}
            color={toGray}
            accent="text-term-cyan"
          />
          <Arrow label="U-Net" />
          <TilePanel
            title="predicted noise ε̂"
            sub={`model output`}
            data={eps_hat}
            color={diffColor}
            accent="text-term-purple"
          />
          <Arrow label="x_t − sqrt(1-ᾱ)·ε̂" />
          <TilePanel
            title="reconstruction x̂₀"
            sub={`(x_t − sqrt(1−ᾱ)·ε̂)/sqrt(ᾱ)`}
            data={residual}
            color={toGray}
            accent="text-term-green"
          />
        </div>
        <div className="flex items-center justify-center gap-6 text-[10px] font-mono text-dark-text-muted pt-1">
          <span>true ε (hidden) — model tries to recover this</span>
          <span className="text-term-amber">|mean ε̂| = {(eps_hat.reduce((a, v) => a + Math.abs(v), 0) / (N * N)).toFixed(2)}</span>
        </div>
      </div>
    </WidgetFrame>
  )
}

function TilePanel({
  title,
  sub,
  data,
  color,
  accent,
}: {
  title: string
  sub: string
  data: number[]
  color: (v: number) => string
  accent: string
}) {
  return (
    <div className="flex flex-col items-center gap-2 min-w-0">
      <div className={`text-[10px] font-mono uppercase tracking-wider ${accent}`}>{title}</div>
      <div
        className="aspect-square w-[110px] rounded-sm border border-dark-border overflow-hidden"
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${N}, 1fr)`,
          gridTemplateRows: `repeat(${N}, 1fr)`,
        }}
      >
        {data.map((v, i) => (
          <div key={i} style={{ backgroundColor: color(v) }} />
        ))}
      </div>
      <div className="text-[9px] font-mono text-dark-text-disabled tabular-nums">{sub}</div>
    </div>
  )
}

function Arrow({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center gap-1 text-dark-text-disabled font-mono text-[9px]">
      <div className="w-10 h-[1px] bg-dark-border" />
      <div>{label}</div>
      <div className="w-0 h-0 border-l-[6px] border-l-dark-border border-y-[4px] border-y-transparent" />
    </div>
  )
}
