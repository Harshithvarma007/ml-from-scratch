'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Side-by-side 2x2 grids of 32x32 tiles.
//   Left  = unconditional samples ε_uncond (pure noise textures)
//   Right = conditional samples under the selected class, computed as
//           ε_cond = class_bias + ε_uncond, then the actual tile displayed is
//           the guided combination ε̂ = ε_uncond + w·(ε_cond − ε_uncond)
// Four class presets give deterministic class biases (circle, square,
// triangle, star). The formula is printed in the readout row.

const N = 32

type ClassName = 'circle' | 'square' | 'triangle' | 'star'
const CLASSES: ClassName[] = ['circle', 'square', 'triangle', 'star']

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

function classBias(kind: ClassName): number[] {
  const out = new Array(N * N).fill(0)
  if (kind === 'circle') {
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++) {
        const d = Math.hypot(x - 16, y - 16)
        out[y * N + x] = d < 8 ? 0.9 : d < 11 ? 0.3 : -0.6
      }
  } else if (kind === 'square') {
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++) {
        out[y * N + x] = x >= 8 && x <= 23 && y >= 8 && y <= 23 ? 0.75 : -0.55
      }
  } else if (kind === 'triangle') {
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++) {
        const row = y - 5
        const half = Math.max(0, row)
        out[y * N + x] = y > 5 && y < 26 && Math.abs(x - 16) <= half && half <= 12 ? 0.8 : -0.55
      }
  } else {
    // star: five-point via radial gate
    for (let y = 0; y < N; y++)
      for (let x = 0; x < N; x++) {
        const dx = x - 16
        const dy = y - 16
        const r = Math.hypot(dx, dy)
        const theta = Math.atan2(dy, dx)
        const rStar = 10 + 4 * Math.cos(5 * theta)
        out[y * N + x] = r < rStar - 2 ? 0.85 : r < rStar + 1 ? 0.2 : -0.55
      }
  }
  return out
}

function sampleTile(seed: number, bias: number[], w: number): number[] {
  // eps_uncond: pure Gaussian-ish texture
  const rng = mulberry32(seed)
  const eps_uncond = new Array(N * N)
  for (let i = 0; i < N * N; i++) eps_uncond[i] = gauss(rng) * 0.7
  // eps_cond = bias + small noise
  const rng2 = mulberry32(seed + 1000)
  const eps_cond = new Array(N * N)
  for (let i = 0; i < N * N; i++) eps_cond[i] = bias[i] + gauss(rng2) * 0.25
  // guided: eps_uncond + w*(eps_cond - eps_uncond)
  const out = new Array(N * N)
  for (let i = 0; i < N * N; i++)
    out[i] = eps_uncond[i] + w * (eps_cond[i] - eps_uncond[i])
  return out
}

function sampleUncond(seed: number): number[] {
  const rng = mulberry32(seed * 7 + 3)
  const out = new Array(N * N)
  for (let i = 0; i < N * N; i++) out[i] = gauss(rng) * 0.7
  return out
}

function toGray(v: number): string {
  const g = Math.max(0, Math.min(255, Math.round((v * 0.5 + 0.5) * 255)))
  return `rgb(${g},${g},${g})`
}

export default function ConditionalVsUnconditional() {
  const [cls, setCls] = useState<ClassName>('circle')
  const [w, setW] = useState(2.5)

  const bias = useMemo(() => classBias(cls), [cls])

  const uncond = useMemo(
    () => [0, 1, 2, 3].map((i) => sampleUncond(i + 1)),
    [],
  )
  const cond = useMemo(
    () => [0, 1, 2, 3].map((i) => sampleTile(i + 10, bias, w)),
    [bias, w],
  )

  return (
    <WidgetFrame
      widgetName="ConditionalVsUnconditional"
      label="unconditional vs class-conditional sampling"
      right={<span className="font-mono">ε̂ = ε_uncond + w·(ε_cond − ε_uncond) · w = {w.toFixed(1)}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            {CLASSES.map((c) => (
              <button
                key={c}
                onClick={() => setCls(c)}
                className={cn(
                  'px-2.5 py-1 rounded text-[10.5px] font-mono uppercase transition-all',
                  cls === c
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {c}
              </button>
            ))}
          </div>
          <Slider label="w" value={w} min={0} max={8} step={0.1} onChange={setW} format={(v) => v.toFixed(1)} accent="accent-term-amber" />
          <div className="flex items-center gap-3 ml-auto">
            <Readout label="class c" value={cls} accent="text-term-purple" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-2 gap-4">
        <GridPanel
          title="unconditional — ε_uncond"
          sub="no class info · diverse · unaligned"
          tiles={uncond}
          titleColor="text-term-rose"
        />
        <GridPanel
          title={`conditional — class = ${cls}`}
          sub={`guided with w = ${w.toFixed(1)}`}
          tiles={cond}
          titleColor="text-term-green"
        />
      </div>
    </WidgetFrame>
  )
}

function GridPanel({
  title,
  sub,
  tiles,
  titleColor,
}: {
  title: string
  sub: string
  tiles: number[][]
  titleColor: string
}) {
  return (
    <div className="flex flex-col gap-2 min-w-0 min-h-0">
      <div className={`text-[10px] font-mono uppercase tracking-wider ${titleColor}`}>
        {title}
      </div>
      <div className="text-[9.5px] font-mono text-dark-text-disabled">{sub}</div>
      <div className="grid grid-cols-2 gap-2 flex-1 min-h-0 p-1">
        {tiles.map((tile, i) => (
          <div
            key={i}
            className="aspect-square rounded border border-dark-border overflow-hidden self-center w-full max-w-[150px] mx-auto"
            style={{
              display: 'grid',
              gridTemplateColumns: `repeat(${N}, 1fr)`,
              gridTemplateRows: `repeat(${N}, 1fr)`,
            }}
          >
            {tile.map((v, j) => (
              <div key={j} style={{ backgroundColor: toGray(v) }} />
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}
