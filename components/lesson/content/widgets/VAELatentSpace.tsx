'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'

// 2D VAE latent space, visualized. 200 deterministic points scattered in
// [-3, 3]^2, each colored by a class derived from its quadrant + angle. The
// user moves a cursor (click or drag on the scatter). The decoder is a
// deterministic function latent(z) → 32x32 tile: we mix the nearest-class
// archetype with a smooth texture modulated by z. The decoded tile updates
// as the cursor moves.

const N_PTS = 200
const GRID = 32

type ClassKey = 'circle' | 'square' | 'triangle' | 'rings'
const CLASSES: ClassKey[] = ['circle', 'square', 'triangle', 'rings']

const CLASS_COLORS: Record<ClassKey, string> = {
  circle: '#67e8f9',
  square: '#fbbf24',
  triangle: '#f472b6',
  rings: '#a78bfa',
}

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

function buildPoints(): { x: number; y: number; cls: ClassKey }[] {
  const rng = mulberry32(101)
  const pts: { x: number; y: number; cls: ClassKey }[] = []
  // Four clusters, one per class
  const CENTERS: { cls: ClassKey; cx: number; cy: number }[] = [
    { cls: 'circle', cx: -1.4, cy: 1.2 },
    { cls: 'square', cx: 1.4, cy: 1.3 },
    { cls: 'triangle', cx: -1.3, cy: -1.3 },
    { cls: 'rings', cx: 1.4, cy: -1.4 },
  ]
  for (let i = 0; i < N_PTS; i++) {
    const c = CENTERS[i % 4]
    pts.push({
      x: c.cx + gauss(rng) * 0.55,
      y: c.cy + gauss(rng) * 0.55,
      cls: c.cls,
    })
  }
  return pts
}

// Archetypes for each class
function archetype(kind: ClassKey): number[] {
  const out = new Array(GRID * GRID).fill(-0.55)
  if (kind === 'circle') {
    for (let y = 0; y < GRID; y++)
      for (let x = 0; x < GRID; x++)
        if (Math.hypot(x - 16, y - 16) < 8) out[y * GRID + x] = 0.85
  } else if (kind === 'square') {
    for (let y = 0; y < GRID; y++)
      for (let x = 0; x < GRID; x++) {
        if (x >= 8 && x <= 23 && y >= 8 && y <= 23) out[y * GRID + x] = 0.75
      }
  } else if (kind === 'triangle') {
    for (let y = 0; y < GRID; y++)
      for (let x = 0; x < GRID; x++) {
        const half = Math.max(0, y - 6)
        if (y > 6 && y < 27 && Math.abs(x - 16) <= half && half <= 11) out[y * GRID + x] = 0.8
      }
  } else {
    for (let y = 0; y < GRID; y++)
      for (let x = 0; x < GRID; x++) {
        const d = Math.hypot(x - 16, y - 16)
        if (d < 4) out[y * GRID + x] = 0.85
        else if (d > 6 && d < 9) out[y * GRID + x] = 0.5
        else if (d > 11 && d < 14) out[y * GRID + x] = 0.2
      }
  }
  return out
}

// Deterministic decoder: given z = (zx, zy), weight the four class archetypes
// by inverse distance to their centers, then modulate with a smooth texture.
const CENTERS: { cls: ClassKey; cx: number; cy: number }[] = [
  { cls: 'circle', cx: -1.4, cy: 1.2 },
  { cls: 'square', cx: 1.4, cy: 1.3 },
  { cls: 'triangle', cx: -1.3, cy: -1.3 },
  { cls: 'rings', cx: 1.4, cy: -1.4 },
]

function decode(zx: number, zy: number, archetypes: Record<ClassKey, number[]>): number[] {
  const weights: Record<ClassKey, number> = { circle: 0, square: 0, triangle: 0, rings: 0 }
  let sum = 0
  CENTERS.forEach((c) => {
    const d = Math.hypot(zx - c.cx, zy - c.cy)
    const w = 1 / (d * d + 0.05)
    weights[c.cls] = w
    sum += w
  })
  ;(Object.keys(weights) as ClassKey[]).forEach((k) => {
    weights[k] /= sum
  })
  const out = new Array(GRID * GRID)
  for (let i = 0; i < GRID * GRID; i++) {
    let v = 0
    ;(Object.keys(weights) as ClassKey[]).forEach((k) => {
      v += weights[k] * archetypes[k][i]
    })
    // Add small position-modulated texture
    const y = Math.floor(i / GRID)
    const x = i % GRID
    const tex = 0.12 * Math.sin(zx * (x - 16) * 0.08 + zy * (y - 16) * 0.08)
    out[i] = v + tex
  }
  return out
}

function toGray(v: number): string {
  const g = Math.max(0, Math.min(255, Math.round((v * 0.5 + 0.5) * 255)))
  return `rgb(${g},${g},${g})`
}

function nearestClass(zx: number, zy: number): ClassKey {
  let best: ClassKey = 'circle'
  let bd = Infinity
  CENTERS.forEach((c) => {
    const d = Math.hypot(zx - c.cx, zy - c.cy)
    if (d < bd) {
      bd = d
      best = c.cls
    }
  })
  return best
}

export default function VAELatentSpace() {
  const [zx, setZx] = useState(0)
  const [zy, setZy] = useState(0)

  const points = useMemo(buildPoints, [])
  const archetypes = useMemo(() => {
    const out: Record<ClassKey, number[]> = {
      circle: archetype('circle'),
      square: archetype('square'),
      triangle: archetype('triangle'),
      rings: archetype('rings'),
    }
    return out
  }, [])
  const decoded = useMemo(() => decode(zx, zy, archetypes), [zx, zy, archetypes])
  const cls = nearestClass(zx, zy)

  const SIZE = 360
  const PAD = 20
  const R = 3
  const toX = (v: number) => PAD + ((v + R) / (2 * R)) * (SIZE - 2 * PAD)
  const toY = (v: number) => SIZE - PAD - ((v + R) / (2 * R)) * (SIZE - 2 * PAD)
  const fromX = (sx: number) => ((sx - PAD) / (SIZE - 2 * PAD)) * (2 * R) - R
  const fromY = (sy: number) => -(((sy - PAD) / (SIZE - 2 * PAD)) * (2 * R) - R)

  function onPointerMove(e: React.PointerEvent<SVGSVGElement>) {
    if ((e.buttons & 1) === 0) return
    const r = e.currentTarget.getBoundingClientRect()
    const sx = ((e.clientX - r.left) / r.width) * SIZE
    const sy = ((e.clientY - r.top) / r.height) * SIZE
    const newX = Math.max(-R, Math.min(R, fromX(sx)))
    const newY = Math.max(-R, Math.min(R, fromY(sy)))
    setZx(newX)
    setZy(newY)
  }

  function onPointerDown(e: React.PointerEvent<SVGSVGElement>) {
    const r = e.currentTarget.getBoundingClientRect()
    const sx = ((e.clientX - r.left) / r.width) * SIZE
    const sy = ((e.clientY - r.top) / r.height) * SIZE
    const newX = Math.max(-R, Math.min(R, fromX(sx)))
    const newY = Math.max(-R, Math.min(R, fromY(sy)))
    setZx(newX)
    setZy(newY)
    e.currentTarget.setPointerCapture(e.pointerId)
  }

  return (
    <WidgetFrame
      widgetName="VAELatentSpace"
      label="2D VAE latent manifold — drag to decode"
      right={<span className="font-mono">z ∈ ℝ² → x̂ · nearest cluster decides class identity</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 font-mono text-[11px] min-w-[160px]">
            <span className="text-dark-text-secondary">z_x</span>
            <input
              type="range"
              min={-R}
              max={R}
              step={0.01}
              value={zx}
              onChange={(e) => setZx(Number(e.target.value))}
              className="flex-1 h-1 rounded-full bg-dark-border accent-term-cyan cursor-pointer"
            />
            <span className="text-dark-text-primary tabular-nums w-10 text-right">{zx.toFixed(2)}</span>
          </label>
          <label className="flex items-center gap-2 font-mono text-[11px] min-w-[160px]">
            <span className="text-dark-text-secondary">z_y</span>
            <input
              type="range"
              min={-R}
              max={R}
              step={0.01}
              value={zy}
              onChange={(e) => setZy(Number(e.target.value))}
              className="flex-1 h-1 rounded-full bg-dark-border accent-term-amber cursor-pointer"
            />
            <span className="text-dark-text-primary tabular-nums w-10 text-right">{zy.toFixed(2)}</span>
          </label>
          <div className="flex items-center gap-3 ml-auto">
            <Readout label="nearest class" value={cls} accent={`text-[${CLASS_COLORS[cls]}]`} />
            <Readout label="‖z‖" value={Math.hypot(zx, zy).toFixed(2)} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-[1fr_auto] gap-4">
        {/* scatter */}
        <div className="flex flex-col gap-1 min-w-0 min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            latent space — click / drag to move z
          </div>
          <div className="flex-1 min-h-0 flex items-center justify-center">
            <svg
              viewBox={`0 0 ${SIZE} ${SIZE}`}
              className="w-full h-full max-h-[300px] cursor-crosshair touch-none"
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
            >
              <rect x={PAD} y={PAD} width={SIZE - 2 * PAD} height={SIZE - 2 * PAD} fill="#0a0a0f" stroke="#1f1f26" />
              {/* axes */}
              <line x1={toX(0)} y1={PAD} x2={toX(0)} y2={SIZE - PAD} stroke="#2a2a32" strokeWidth={1} />
              <line x1={PAD} y1={toY(0)} x2={SIZE - PAD} y2={toY(0)} stroke="#2a2a32" strokeWidth={1} />
              {/* points */}
              {points.map((p, i) => (
                <circle
                  key={i}
                  cx={toX(p.x)}
                  cy={toY(p.y)}
                  r={2.3}
                  fill={CLASS_COLORS[p.cls]}
                  fillOpacity={0.65}
                />
              ))}
              {/* cursor */}
              <circle
                cx={toX(zx)}
                cy={toY(zy)}
                r={9}
                fill="none"
                stroke="#fff"
                strokeWidth={1.4}
                strokeDasharray="3 3"
              />
              <circle cx={toX(zx)} cy={toY(zy)} r={3.2} fill="#fff" />
            </svg>
          </div>
          {/* legend */}
          <div className="flex items-center justify-center gap-4 text-[10px] font-mono">
            {CLASSES.map((c) => (
              <span key={c} className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: CLASS_COLORS[c] }} />
                <span className="text-dark-text-secondary">{c}</span>
              </span>
            ))}
          </div>
        </div>

        {/* decoded tile */}
        <div className="flex flex-col gap-2 items-center min-w-[160px]">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            decoder(z) → x̂
          </div>
          <div
            className="aspect-square w-[160px] rounded border overflow-hidden"
            style={{
              display: 'grid',
              gridTemplateColumns: `repeat(${GRID}, 1fr)`,
              gridTemplateRows: `repeat(${GRID}, 1fr)`,
              borderColor: CLASS_COLORS[cls],
              boxShadow: `0 0 0 2px ${CLASS_COLORS[cls]}33`,
            }}
          >
            {decoded.map((v, i) => (
              <div key={i} style={{ backgroundColor: toGray(v) }} />
            ))}
          </div>
          <div className="text-[10px] font-mono text-dark-text-muted text-center leading-snug max-w-[160px]">
            smooth interpolation between clusters — move between two colors to watch one
            shape morph into the other.
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
