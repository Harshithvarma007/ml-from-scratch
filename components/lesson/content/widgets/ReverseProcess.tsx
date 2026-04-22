'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { Play, Pause, SkipBack, StepForward, StepBack } from 'lucide-react'

// Side-by-side reverse process: starting from x_T ~ N(0, I) for 200 points, we
// step back through time T-1, T-2, ..., 0 using an *oracle* denoiser that
// pulls each noisy point toward its known x_0 (a ring sample). The left panel
// shows the forward ring as reference; the right panel shows x_t evolving.
// The slider or play button animates the reverse trajectory.

const N_POINTS = 200
const T = 60
const BETA_START = 1e-4
const BETA_END = 0.035

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

function buildSchedule() {
  const betas: number[] = []
  const alphaBar: number[] = []
  let prod = 1
  for (let t = 0; t < T; t++) {
    const b = BETA_START + (BETA_END - BETA_START) * (t / (T - 1))
    betas.push(b)
    prod *= 1 - b
    alphaBar.push(prod)
  }
  return { betas, alphaBar }
}

// Precompute trajectories: for each point we already know x_0 (the ring
// sample), and we synthesise x_T by adding noise. The reverse walk then
// uses DDIM's deterministic update:
//    x_{t-1} = sqrt(ab_{t-1}) * x_0_hat + sqrt(1 - ab_{t-1}) * eps_hat
// where x_0_hat = x_0 (oracle) and eps_hat is recovered from x_t.
function buildTrajectories() {
  const x0 = ringPoints()
  const { alphaBar } = buildSchedule()
  const rng = mulberry32(99)

  // At each t, store positions for all N_POINTS.
  const trajectory: { x: number; y: number }[][] = []
  for (let t = 0; t <= T; t++) trajectory.push([])

  // sample x_T for each point
  const x_T = x0.map((p) => {
    const ab = alphaBar[T - 1]
    return {
      x: Math.sqrt(ab) * p.x + Math.sqrt(1 - ab) * gauss(rng),
      y: Math.sqrt(ab) * p.y + Math.sqrt(1 - ab) * gauss(rng),
    }
  })

  // Walk back using DDIM-oracle step:
  //   eps = (x_t - sqrt(ab_t) * x0) / sqrt(1 - ab_t)
  //   x_{t-1} = sqrt(ab_{t-1}) * x0 + sqrt(1 - ab_{t-1}) * eps
  // We store x_t at every step t.
  let current = x_T
  trajectory[T] = current
  for (let t = T - 1; t >= 0; t--) {
    const ab_t = alphaBar[t]
    const ab_prev = t === 0 ? 1.0 : alphaBar[t - 1]
    const next: { x: number; y: number }[] = current.map((p, i) => {
      const x0_p = x0[i]
      const epsX = (p.x - Math.sqrt(ab_t) * x0_p.x) / Math.sqrt(1 - ab_t)
      const epsY = (p.y - Math.sqrt(ab_t) * x0_p.y) / Math.sqrt(1 - ab_t)
      return {
        x: Math.sqrt(ab_prev) * x0_p.x + Math.sqrt(1 - ab_prev) * epsX,
        y: Math.sqrt(ab_prev) * x0_p.y + Math.sqrt(1 - ab_prev) * epsY,
      }
    })
    trajectory[t] = next
    current = next
  }
  return { x0, trajectory, alphaBar }
}

export default function ReverseProcess() {
  const [t, setT] = useState(T)
  const [playing, setPlaying] = useState(false)
  const rafRef = useRef<number | null>(null)

  const { x0, trajectory, alphaBar } = useMemo(buildTrajectories, [])
  const x_t = trajectory[t] ?? trajectory[0]

  // Forward panel: show the ring + current noised version (forward at t).
  // We reuse trajectory[t] for the "noised" side since it equals q(x_t|x_0)
  // by construction (oracle).
  useEffect(() => {
    if (!playing) return
    const tick = () => {
      setT((prev) => {
        if (prev <= 0) {
          setPlaying(false)
          return 0
        }
        return prev - 1
      })
      rafRef.current = window.setTimeout(tick, 80) as unknown as number
    }
    rafRef.current = window.setTimeout(tick, 80) as unknown as number
    return () => {
      if (rafRef.current) window.clearTimeout(rafRef.current)
    }
  }, [playing])

  const ab = alphaBar[Math.min(t, T - 1)]
  const var_t = 1 - ab

  return (
    <WidgetFrame
      widgetName="ReverseProcess"
      label="reverse process — Gaussian blob re-collapses into a ring"
      right={<span className="font-mono">x_(t−1) = sqrt(ᾱ_(t−1))·x₀ + sqrt(1−ᾱ_(t−1))·ε̂</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={() => { setT(T); setPlaying(false) }} variant="ghost">
            <span className="inline-flex items-center gap-1"><SkipBack size={11} /> reset</span>
          </Button>
          <Button onClick={() => setT(Math.min(T, t + 1))} variant="ghost" disabled={t >= T}>
            <span className="inline-flex items-center gap-1"><StepBack size={11} /> +1</span>
          </Button>
          <Button onClick={() => setT(Math.max(0, t - 1))} variant="ghost" disabled={t <= 0}>
            <span className="inline-flex items-center gap-1">−1 <StepForward size={11} /></span>
          </Button>
          <Button onClick={() => setPlaying((p) => !p)} variant="primary" disabled={t === 0 && !playing}>
            <span className="inline-flex items-center gap-1">
              {playing ? (<><Pause size={11} /> pause</>) : (<><Play size={11} /> play reverse</>)}
            </span>
          </Button>
          <input
            type="range"
            min={0}
            max={T}
            step={1}
            value={t}
            onChange={(e) => { setT(Number(e.target.value)); setPlaying(false) }}
            className="flex-1 min-w-[120px] h-1 rounded-full bg-dark-border accent-term-pink cursor-pointer"
          />
          <div className="flex items-center gap-3 ml-auto">
            <Readout label="t" value={String(t)} accent="text-term-pink" />
            <Readout label="ᾱ_t" value={ab.toFixed(3)} accent="text-term-amber" />
            <Readout label="var" value={var_t.toFixed(3)} accent="text-term-cyan" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-2 gap-4">
        <Panel
          title="forward reference"
          sub="q(x_t | x_0)"
          x0={x0}
          live={trajectory[t] ?? []}
          liveColor="rgba(103, 232, 249, 0.8)"
          titleColor="text-term-cyan"
        />
        <Panel
          title="reverse trajectory"
          sub="x_T → x_(t) → x_0"
          x0={x0}
          live={x_t}
          liveColor="rgba(244, 114, 182, 0.85)"
          titleColor="text-term-pink"
        />
      </div>
    </WidgetFrame>
  )
}

function Panel({
  title,
  sub,
  x0,
  live,
  liveColor,
  titleColor,
}: {
  title: string
  sub: string
  x0: { x: number; y: number }[]
  live: { x: number; y: number }[]
  liveColor: string
  titleColor: string
}) {
  const SIZE = 360
  const PAD = 18
  const R = 3.2
  const toX = (v: number) => PAD + ((v + R) / (2 * R)) * (SIZE - 2 * PAD)
  const toY = (v: number) => SIZE - PAD - ((v + R) / (2 * R)) * (SIZE - 2 * PAD)

  return (
    <div className="flex flex-col gap-1 min-w-0 min-h-0">
      <div className={`text-[10px] font-mono uppercase tracking-wider ${titleColor}`}>
        {title} <span className="text-dark-text-disabled normal-case"> · {sub}</span>
      </div>
      <div className="flex-1 min-h-0 flex items-center justify-center">
        <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="w-full h-full max-h-[280px]">
          <line x1={toX(0)} y1={PAD} x2={toX(0)} y2={SIZE - PAD} stroke="#1f1f26" strokeWidth={1} />
          <line x1={PAD} y1={toY(0)} x2={SIZE - PAD} y2={toY(0)} stroke="#1f1f26" strokeWidth={1} />
          <rect
            x={PAD}
            y={PAD}
            width={SIZE - 2 * PAD}
            height={SIZE - 2 * PAD}
            fill="none"
            stroke="#1f1f26"
            strokeWidth={1}
          />
          {x0.map((p, i) => (
            <circle key={`x0-${i}`} cx={toX(p.x)} cy={toY(p.y)} r={1.3} fill="rgba(74, 222, 128, 0.2)" />
          ))}
          {live.map((p, i) => (
            <circle key={`live-${i}`} cx={toX(p.x)} cy={toY(p.y)} r={2.2} fill={liveColor} />
          ))}
        </svg>
      </div>
    </div>
  )
}
