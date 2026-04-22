'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { Dice5, StepForward } from 'lucide-react'

// Walks through one DDPM training step:
//   1. sample x_0 from data (deterministic scene)
//   2. sample t in [0, T-1]
//   3. sample eps ~ N(0, I)
//   4. x_t = sqrt(ab) * x_0 + sqrt(1-ab) * eps
//   5. model predicts eps_hat (noisier copy of eps controlled by "model error")
//   6. loss = MSE(eps, eps_hat)
// Shows all five visual tensors at 32x32 plus a loss bar. "new batch" re-seeds.

const N = 32
const T = 100
const BETA_START = 1e-4
const BETA_END = 0.02

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

function dataImage(seed: number): number[] {
  // Pick one of four deterministic scenes by seed
  const kind = seed % 4
  const out = new Array(N * N).fill(-0.5)
  if (kind === 0) {
    // circle
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++)
        if (Math.hypot(x - 16, y - 16) < 9) out[y * N + x] = 0.8
  } else if (kind === 1) {
    // square
    for (let y = 8; y < 24; y++) for (let x = 8; x < 24; x++) out[y * N + x] = 0.7
  } else if (kind === 2) {
    // diag stripe
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++)
        if (Math.abs(x - y) < 3) out[y * N + x] = 0.75
  } else {
    // cross
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++)
        if (Math.abs(x - 16) < 3 || Math.abs(y - 16) < 3) out[y * N + x] = 0.7
  }
  return out
}

function alphaBarAt(t: number): number {
  let prod = 1
  for (let i = 0; i <= t; i++) {
    const b = BETA_START + (BETA_END - BETA_START) * (i / (T - 1))
    prod *= 1 - b
  }
  return prod
}

function toGray(v: number): string {
  const g = Math.max(0, Math.min(255, Math.round((v * 0.5 + 0.5) * 255)))
  return `rgb(${g},${g},${g})`
}

function diffColor(v: number): string {
  const a = Math.min(1, Math.abs(v) / 2)
  if (v >= 0) return `rgba(103, 232, 249, ${0.1 + a * 0.85})`
  return `rgba(251, 113, 133, ${0.1 + a * 0.85})`
}

export default function DDPMTraining() {
  const [batchSeed, setBatchSeed] = useState(1)
  const [modelError, setModelError] = useState(0.45)

  const step = useMemo(() => {
    const rng = mulberry32(batchSeed * 1337)
    // random t
    const t = Math.floor(rng() * T)
    const ab = alphaBarAt(t)
    const x0 = dataImage(batchSeed + 3)
    const eps = new Array(N * N)
    for (let i = 0; i < N * N; i++) eps[i] = gauss(rng)
    const x_t = new Array(N * N)
    for (let i = 0; i < N * N; i++)
      x_t[i] = Math.sqrt(ab) * x0[i] + Math.sqrt(1 - ab) * eps[i]
    // eps_hat: fraction of true eps + fraction of noise
    const eps_hat = new Array(N * N)
    const rng2 = mulberry32(batchSeed * 999 + 17)
    for (let i = 0; i < N * N; i++) {
      eps_hat[i] = (1 - modelError) * eps[i] + modelError * gauss(rng2)
    }
    let mse = 0
    for (let i = 0; i < N * N; i++) {
      const d = eps_hat[i] - eps[i]
      mse += d * d
    }
    mse /= N * N
    return { t, ab, x0, eps, x_t, eps_hat, mse }
  }, [batchSeed, modelError])

  return (
    <WidgetFrame
      widgetName="DDPMTraining"
      label="one DDPM training iteration — end-to-end"
      right={<span className="font-mono">loss = MSE(ε̂, ε) · one step of SGD · T = {T}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={() => setBatchSeed((s) => s + 1)} variant="primary">
            <span className="inline-flex items-center gap-1">
              <Dice5 size={11} /> sample new batch
            </span>
          </Button>
          <Button onClick={() => setModelError((e) => Math.max(0, e - 0.1))} variant="ghost">
            <span className="inline-flex items-center gap-1">
              <StepForward size={11} /> better model
            </span>
          </Button>
          <label className="flex items-center gap-2 font-mono text-[11px] min-w-[180px]">
            <span className="text-dark-text-secondary">model err</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={modelError}
              onChange={(e) => setModelError(Number(e.target.value))}
              className="flex-1 h-1 rounded-full bg-dark-border accent-term-amber cursor-pointer"
            />
            <span className="text-dark-text-primary tabular-nums w-10 text-right">{modelError.toFixed(2)}</span>
          </label>
          <div className="flex items-center gap-3 ml-auto">
            <Readout label="t" value={String(step.t)} accent="text-term-purple" />
            <Readout label="ᾱ_t" value={step.ab.toFixed(3)} accent="text-term-amber" />
            <Readout
              label="loss"
              value={step.mse.toFixed(3)}
              accent={step.mse < 0.3 ? 'text-term-green' : step.mse > 0.8 ? 'text-term-rose' : 'text-term-amber'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden flex flex-col gap-3">
        <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
          training step: sample · noise · predict · loss
        </div>
        <div className="flex-1 min-h-0 flex items-center justify-between gap-3">
          <Tile title="1. x₀" sub="clean sample" data={step.x0} color={toGray} accent="text-term-green" />
          <Plus />
          <Tile title="2. ε" sub="N(0, I)" data={step.eps} color={diffColor} accent="text-term-purple" />
          <Eq label="=" />
          <Tile
            title={`3. x_${step.t}`}
            sub={`sqrt(ᾱ)x₀ + sqrt(1−ᾱ)ε`}
            data={step.x_t}
            color={toGray}
            accent="text-term-cyan"
          />
          <Arrow label="U-Net(x_t, t)" />
          <Tile title="4. ε̂" sub="model output" data={step.eps_hat} color={diffColor} accent="text-term-amber" />
          <div className="flex flex-col gap-1 items-center font-mono text-[10px]">
            <span className="text-term-rose">loss</span>
            <div className="relative h-20 w-4 bg-dark-surface-elevated/40 rounded overflow-hidden border border-dark-border">
              <div
                className="absolute bottom-0 left-0 right-0 bg-term-rose/80"
                style={{ height: `${Math.min(100, step.mse * 80)}%` }}
              />
            </div>
            <span className="text-dark-text-primary tabular-nums">{step.mse.toFixed(2)}</span>
          </div>
        </div>
        <div className="text-[10.5px] font-mono text-dark-text-muted leading-snug">
          <span className="text-term-green">forward</span> fixes what you feed the model ·{' '}
          <span className="text-term-amber">backward</span> is one SGD step on ‖ε̂ − ε‖² ·
          repeat millions of times
        </div>
      </div>
    </WidgetFrame>
  )
}

function Tile({
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
    <div className="flex flex-col items-center gap-1 min-w-0">
      <div className={`text-[10px] font-mono uppercase tracking-wider ${accent}`}>{title}</div>
      <div
        className="aspect-square w-[82px] rounded-sm border border-dark-border overflow-hidden"
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
      <div className="text-[9px] font-mono text-dark-text-disabled">{sub}</div>
    </div>
  )
}

function Plus() {
  return <div className="text-dark-text-disabled font-mono text-sm">+</div>
}
function Eq({ label }: { label: string }) {
  return <div className="text-dark-text-disabled font-mono text-sm">{label}</div>
}
function Arrow({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center font-mono text-[9px] text-dark-text-muted">
      <div className="w-10 h-[1px] bg-dark-border" />
      <span>{label}</span>
      <div className="w-0 h-0 border-l-[5px] border-l-dark-border border-y-[3px] border-y-transparent" />
    </div>
  )
}
