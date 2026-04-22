'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Ten 32x32 tiles along a noise staircase. The clean image (t=0) is a
// deterministic geometric scene — a filled circle and a square on a dark
// background. Each subsequent tile shows the same scene re-sampled with
// x_t = sqrt(1 - beta_t) * x_0 + sqrt(beta_t) * eps using a linear beta
// schedule. The slider picks the highlighted step; the overlay calls out
// "clean" at t=0 and "pure noise" at t=9.

const STEPS = 10
const N = 32
const BETA_START = 0.04
const BETA_END = 0.55

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

// Deterministic clean "image": a bright circle and a bright square on a dim
// canvas. Values live in [-1, 1] so the diffusion math is symmetric.
function cleanImage(): number[] {
  const out = new Array(N * N).fill(-0.6)
  const cx1 = 10
  const cy1 = 11
  const r = 6
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const d = Math.hypot(x - cx1, y - cy1)
      if (d < r) out[y * N + x] = 0.9 - (d / r) * 0.2
    }
  }
  // square
  for (let y = 17; y < 28; y++) {
    for (let x = 19; x < 29; x++) {
      out[y * N + x] = 0.6
    }
  }
  return out
}

function tilesAt(x0: number[]): { tiles: number[][]; betas: number[]; alphaBar: number[] } {
  const betas: number[] = []
  for (let t = 0; t < STEPS; t++) {
    betas.push(BETA_START + (BETA_END - BETA_START) * (t / (STEPS - 1)))
  }
  const alphas = betas.map((b) => 1 - b)
  const alphaBar: number[] = []
  let prod = 1
  for (let t = 0; t < STEPS; t++) {
    prod *= alphas[t]
    alphaBar.push(prod)
  }
  const tiles: number[][] = []
  for (let t = 0; t < STEPS; t++) {
    const rng = mulberry32(7 * t + 3)
    const ab = alphaBar[t]
    const tile = new Array(N * N)
    for (let i = 0; i < N * N; i++) {
      const eps = gauss(rng)
      tile[i] = Math.sqrt(ab) * x0[i] + Math.sqrt(1 - ab) * eps
    }
    tiles.push(tile)
  }
  return { tiles, betas, alphaBar }
}

function toGray(v: number): string {
  // Map [-1, 1] to [0, 255]
  const g = Math.max(0, Math.min(255, Math.round((v * 0.5 + 0.5) * 255)))
  return `rgb(${g},${g},${g})`
}

function TileGrid({ data, size, highlight }: { data: number[]; size: number; highlight?: boolean }) {
  return (
    <div
      className={cn(
        'rounded-sm overflow-hidden border transition-all',
        highlight ? 'border-term-amber shadow-[0_0_0_1px_rgba(251,191,36,0.6)]' : 'border-dark-border',
      )}
      style={{
        width: size,
        height: size,
        display: 'grid',
        gridTemplateColumns: `repeat(${N}, 1fr)`,
        gridTemplateRows: `repeat(${N}, 1fr)`,
      }}
    >
      {data.map((v, i) => (
        <div key={i} style={{ backgroundColor: toGray(v) }} />
      ))}
    </div>
  )
}

export default function NoiseStaircase() {
  const x0 = useMemo(() => cleanImage(), [])
  const { tiles, betas, alphaBar } = useMemo(() => tilesAt(x0), [x0])
  const [t, setT] = useState(4)

  const beta_t = betas[t]
  const ab = alphaBar[t]

  return (
    <WidgetFrame
      widgetName="NoiseStaircase"
      label="forward diffusion staircase — clean to pure noise in 10 steps"
      right={<span className="font-mono">x_t = sqrt(1-β_t)·x_(t-1) + sqrt(β_t)·ε</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="t"
            value={t}
            min={0}
            max={STEPS - 1}
            step={1}
            onChange={(v) => setT(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="β_t" value={beta_t.toFixed(3)} accent="text-term-cyan" />
            <Readout label="ᾱ_t" value={ab.toFixed(3)} accent="text-term-amber" />
            <Readout label="snr" value={(ab / Math.max(1 - ab, 1e-6)).toFixed(2)} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden flex flex-col gap-3">
        <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
          x_0  —  x_1  —  x_2  —  ···  —  x_9
        </div>
        <div className="flex-1 min-h-0 flex items-center">
          <div className="flex items-end justify-between gap-1.5 w-full">
            {tiles.map((tile, i) => (
              <div key={i} className="flex flex-col items-center gap-1.5 flex-1 min-w-0">
                <div className="text-[9px] font-mono text-dark-text-disabled tabular-nums">
                  t={i}
                </div>
                <div className="flex items-center justify-center w-full">
                  <TileGridResponsive data={tile} highlight={i === t} />
                </div>
                <div
                  className={cn(
                    'text-[9px] font-mono tabular-nums',
                    i === t ? 'text-term-amber' : 'text-dark-text-muted',
                  )}
                >
                  β={betas[i].toFixed(2)}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="flex items-center justify-between text-[10px] font-mono text-dark-text-muted">
          <span className="text-term-green">◄ clean (signal-dominated)</span>
          <span className="text-term-rose">pure noise (variance ≈ 1) ►</span>
        </div>
      </div>
    </WidgetFrame>
  )
}

// Responsive tile: fills its flex box up to ~70px.
function TileGridResponsive({ data, highlight }: { data: number[]; highlight: boolean }) {
  return (
    <div
      className={cn(
        'rounded-sm overflow-hidden border aspect-square w-full max-w-[72px]',
        highlight ? 'border-term-amber' : 'border-dark-border',
      )}
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${N}, 1fr)`,
        gridTemplateRows: `repeat(${N}, 1fr)`,
        boxShadow: highlight ? '0 0 0 2px rgba(251,191,36,0.35)' : undefined,
      }}
    >
      {data.map((v, i) => (
        <div key={i} style={{ backgroundColor: toGray(v) }} />
      ))}
    </div>
  )
}
