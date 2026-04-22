'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// 2D scatter of 200 points that start on a ring distribution (x_0) and
// progressively become a standard Gaussian blob (x_T). We apply the closed-
// form q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) I) with a
// linear beta schedule. The slider picks t; the plot shows both x_0 (ghost)
// and x_t (live). A side column shows the schedule values at t.

const N_POINTS = 200
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

function buildSchedule() {
  const betas: number[] = []
  const alphas: number[] = []
  const alphaBar: number[] = []
  let prod = 1
  for (let t = 0; t < T; t++) {
    const b = BETA_START + (BETA_END - BETA_START) * (t / (T - 1))
    betas.push(b)
    alphas.push(1 - b)
    prod *= 1 - b
    alphaBar.push(prod)
  }
  return { betas, alphas, alphaBar }
}

function ringPoints(): { x: number; y: number }[] {
  const rng = mulberry32(42)
  const pts: { x: number; y: number }[] = []
  for (let i = 0; i < N_POINTS; i++) {
    const theta = (i / N_POINTS) * Math.PI * 2 + rng() * 0.2
    const r = 1.6 + (rng() - 0.5) * 0.12
    pts.push({ x: r * Math.cos(theta), y: r * Math.sin(theta) })
  }
  return pts
}

function forwardSample(
  pts: { x: number; y: number }[],
  alphaBar: number,
  seed: number,
): { x: number; y: number }[] {
  const rng = mulberry32(seed)
  const s = Math.sqrt(alphaBar)
  const n = Math.sqrt(1 - alphaBar)
  return pts.map((p) => ({
    x: s * p.x + n * gauss(rng),
    y: s * p.y + n * gauss(rng),
  }))
}

export default function ForwardProcess() {
  const [t, setT] = useState(40)
  const schedule = useMemo(buildSchedule, [])
  const x0 = useMemo(ringPoints, [])
  const x_t = useMemo(
    () => forwardSample(x0, schedule.alphaBar[t], 2024 + t),
    [x0, schedule, t],
  )

  const ab = schedule.alphaBar[t]
  const beta_t = schedule.betas[t]

  return (
    <WidgetFrame
      widgetName="ForwardProcess"
      label="forward process — ring dissolves into Gaussian noise"
      right={<span className="font-mono">q(x_t|x_0) = N(sqrt(ᾱ_t)·x_0, (1−ᾱ_t)·I)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="t"
            value={t}
            min={0}
            max={T - 1}
            step={1}
            onChange={(v) => setT(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-cyan"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="β_t" value={beta_t.toExponential(2)} />
            <Readout label="ᾱ_t" value={ab.toFixed(3)} accent="text-term-amber" />
            <Readout label="sqrt(ᾱ)" value={Math.sqrt(ab).toFixed(3)} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-1 md:grid-cols-[1fr_220px] gap-4">
        <ScatterCanvas x0={x0} x_t={x_t} t={t} />
        <SchedulePanel betas={schedule.betas} alphaBar={schedule.alphaBar} t={t} />
      </div>
    </WidgetFrame>
  )
}

function ScatterCanvas({
  x0,
  x_t,
  t,
}: {
  x0: { x: number; y: number }[]
  x_t: { x: number; y: number }[]
  t: number
}) {
  const SIZE = 420
  const PAD = 20
  const R = 3.2
  const toX = (v: number) => PAD + ((v + R) / (2 * R)) * (SIZE - 2 * PAD)
  const toY = (v: number) => SIZE - PAD - ((v + R) / (2 * R)) * (SIZE - 2 * PAD)

  return (
    <div className="relative w-full h-full min-w-0 flex items-center justify-center">
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="w-full h-full max-h-[320px]">
        {[-3, -2, -1, 0, 1, 2, 3].map((g) => (
          <g key={g}>
            <line
              x1={toX(g)}
              y1={PAD}
              x2={toX(g)}
              y2={SIZE - PAD}
              stroke="#1a1a1f"
              strokeWidth={0.7}
            />
            <line
              x1={PAD}
              y1={toY(g)}
              x2={SIZE - PAD}
              y2={toY(g)}
              stroke="#1a1a1f"
              strokeWidth={0.7}
            />
          </g>
        ))}
        <line x1={toX(0)} y1={PAD} x2={toX(0)} y2={SIZE - PAD} stroke="#2a2a32" strokeWidth={1} />
        <line x1={PAD} y1={toY(0)} x2={SIZE - PAD} y2={toY(0)} stroke="#2a2a32" strokeWidth={1} />

        {x0.map((p, i) => (
          <circle
            key={`ghost-${i}`}
            cx={toX(p.x)}
            cy={toY(p.y)}
            r={1.5}
            fill="rgba(74, 222, 128, 0.18)"
          />
        ))}
        {x_t.map((p, i) => (
          <circle key={`xt-${i}`} cx={toX(p.x)} cy={toY(p.y)} r={2.2} fill="rgba(103, 232, 249, 0.85)" />
        ))}

        <text x={PAD + 6} y={PAD + 12} fontSize="10" fill="#67e8f9" fontFamily="JetBrains Mono, monospace">
          x_t  (t = {t})
        </text>
        <text x={PAD + 6} y={PAD + 28} fontSize="10" fill="#4ade80" fontFamily="JetBrains Mono, monospace">
          x_0  (ghost ring)
        </text>
      </svg>
    </div>
  )
}

function SchedulePanel({
  betas,
  alphaBar,
  t,
}: {
  betas: number[]
  alphaBar: number[]
  t: number
}) {
  const W = 220
  const H = 100
  const toX = (i: number) => (i / (T - 1)) * W
  const toY = (v: number, min: number, max: number) =>
    H - 8 - ((v - min) / (max - min)) * (H - 16)
  const betaPath = betas
    .map((b, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toY(b, 0, BETA_END).toFixed(1)}`)
    .join(' ')
  const abPath = alphaBar
    .map((a, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toY(a, 0, 1).toFixed(1)}`)
    .join(' ')

  return (
    <div className="flex flex-col gap-3 min-w-0">
      <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
        schedule
      </div>
      <div className="flex flex-col gap-2 bg-dark-surface-elevated/40 rounded p-2">
        <div className="flex items-center justify-between text-[10px] font-mono">
          <span className="text-term-rose">β_t (linear)</span>
          <span className="text-dark-text-muted tabular-nums">{betas[t].toExponential(2)}</span>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[60px]">
          <path d={betaPath} stroke="#fb7185" strokeWidth={1.5} fill="none" />
          <line x1={toX(t)} y1={0} x2={toX(t)} y2={H} stroke="rgba(255,255,255,0.3)" strokeDasharray="2 3" />
          <circle cx={toX(t)} cy={toY(betas[t], 0, BETA_END)} r={3} fill="#fb7185" />
        </svg>
      </div>

      <div className="flex flex-col gap-2 bg-dark-surface-elevated/40 rounded p-2">
        <div className="flex items-center justify-between text-[10px] font-mono">
          <span className="text-term-amber">ᾱ_t = Π(1−β_s)</span>
          <span className="text-dark-text-muted tabular-nums">{alphaBar[t].toFixed(3)}</span>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[60px]">
          <path d={abPath} stroke="#fbbf24" strokeWidth={1.5} fill="none" />
          <line x1={toX(t)} y1={0} x2={toX(t)} y2={H} stroke="rgba(255,255,255,0.3)" strokeDasharray="2 3" />
          <circle cx={toX(t)} cy={toY(alphaBar[t], 0, 1)} r={3} fill="#fbbf24" />
        </svg>
      </div>

      <div className="text-[10.5px] font-mono text-dark-text-muted leading-snug">
        <div className="text-term-cyan">signal scale: {Math.sqrt(alphaBar[t]).toFixed(3)}</div>
        <div className="text-term-purple">noise scale:  {Math.sqrt(1 - alphaBar[t]).toFixed(3)}</div>
      </div>
    </div>
  )
}
