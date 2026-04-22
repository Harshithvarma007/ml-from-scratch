'use client'

import { useEffect, useMemo, useRef } from 'react'
import WidgetFrame from './WidgetFrame'
import { cn } from '@/lib/utils'

// Three procedurally generated training curves on the same seed. Each curve
// starts near zero, climbs toward optimum 1.0, but the noise envelope differs:
// REINFORCE keeps its ~0.35 σ, baseline halves it, advantage halves it again.
// Under the curves, we show metric-compare bars for final return, episodes to
// 80% of optimum, and variance of gradient estimator (synthetic).

const N_EPISODES = 500

function mulberry32(seed: number) {
  let s = seed
  return () => {
    let t = (s += 0x6d2b79f5)
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

// Variant keys. Each has a noise scale and a learning-speed modifier because
// lower variance → faster convergence in expectation.
const VARIANTS = [
  { key: 'reinforce', label: 'REINFORCE', color: '#f472b6', noise: 0.42, speed: 1.0 },
  { key: 'baseline', label: 'REINFORCE + baseline', color: '#67e8f9', noise: 0.2, speed: 1.35 },
  { key: 'advantage', label: 'REINFORCE + advantage', color: '#4ade80', noise: 0.11, speed: 1.7 },
] as const

type VKey = (typeof VARIANTS)[number]['key']

function simulate(seed: number): Record<VKey, number[]> {
  const out: Record<VKey, number[]> = { reinforce: [], baseline: [], advantage: [] }
  VARIANTS.forEach((v) => {
    // Share the base rng so runs align in their random walk — then shrink by noise scale.
    const rng = mulberry32(seed * 17 + v.noise * 1000)
    for (let t = 0; t < N_EPISODES; t++) {
      const mean = 1 - Math.exp(-(t * v.speed) / 160)
      const noise = gauss(rng) * Math.max(0.05, v.noise * Math.exp(-t / 500))
      out[v.key].push(Math.max(-0.6, Math.min(1.3, mean + noise)))
    }
  })
  return out
}

function rollingMean(data: number[], w: number): number[] {
  const out = new Array(data.length).fill(0)
  for (let i = 0; i < data.length; i++) {
    const lo = Math.max(0, i - w + 1)
    const slice = data.slice(lo, i + 1)
    out[i] = slice.reduce((a, v) => a + v, 0) / slice.length
  }
  return out
}

function rollingStd(data: number[], w: number): number[] {
  const out = new Array(data.length).fill(0)
  for (let i = 0; i < data.length; i++) {
    const lo = Math.max(0, i - w + 1)
    const slice = data.slice(lo, i + 1)
    const m = slice.reduce((a, v) => a + v, 0) / slice.length
    out[i] = Math.sqrt(slice.reduce((a, v) => a + (v - m) ** 2, 0) / slice.length)
  }
  return out
}

function episodesToThreshold(smooth: number[], thr: number): number {
  for (let i = 0; i < smooth.length; i++) if (smooth[i] >= thr) return i
  return smooth.length
}

export default function VarianceReductionComparison() {
  const runs = useMemo(() => simulate(13), [])
  const smoothed = useMemo(() => {
    const out: Record<VKey, number[]> = { reinforce: [], baseline: [], advantage: [] }
    VARIANTS.forEach((v) => (out[v.key] = rollingMean(runs[v.key], 30)))
    return out
  }, [runs])
  const stds = useMemo(() => {
    const out: Record<VKey, number[]> = { reinforce: [], baseline: [], advantage: [] }
    VARIANTS.forEach((v) => (out[v.key] = rollingStd(runs[v.key], 30)))
    return out
  }, [runs])

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const box = boxRef.current
    if (!canvas || !box) return
    const dpr = window.devicePixelRatio || 1
    const draw = () => {
      const w = box.clientWidth
      const h = box.clientHeight
      canvas.width = w * dpr
      canvas.height = h * dpr
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      const ctx = canvas.getContext('2d')!
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, w, h)
      const padL = 42, padR = 14, padT = 14, padB = 24
      const plotW = w - padL - padR
      const plotH = h - padT - padB
      const yMin = -0.5
      const yMax = 1.3
      const toSx = (t: number) => padL + (t / (N_EPISODES - 1)) * plotW
      const toSy = (v: number) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH

      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.strokeStyle = '#1e1e1e'
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      ;[-0.5, 0, 0.5, 1].forEach((y) => {
        const sy = toSy(y)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(w - padR, sy)
        ctx.stroke()
        ctx.fillText(y.toFixed(1), padL - 6, sy + 3)
      })
      ctx.textAlign = 'center'
      ;[0, 100, 200, 300, 400, 499].forEach((t) => ctx.fillText(String(t), toSx(t), padT + plotH + 14))

      // Bands
      VARIANTS.forEach((v) => {
        ctx.fillStyle = v.color + '22'
        ctx.beginPath()
        for (let t = 0; t < N_EPISODES; t++) {
          const sx = toSx(t)
          const sy = toSy(Math.min(yMax, smoothed[v.key][t] + stds[v.key][t]))
          if (t === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        for (let t = N_EPISODES - 1; t >= 0; t--) {
          const sx = toSx(t)
          const sy = toSy(Math.max(yMin, smoothed[v.key][t] - stds[v.key][t]))
          ctx.lineTo(sx, sy)
        }
        ctx.closePath()
        ctx.fill()
      })

      // Means
      VARIANTS.forEach((v) => {
        ctx.strokeStyle = v.color
        ctx.lineWidth = 2.2
        ctx.beginPath()
        for (let t = 0; t < N_EPISODES; t++) {
          const sx = toSx(t)
          const sy = toSy(smoothed[v.key][t])
          if (t === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        ctx.stroke()
      })

      // Legend
      let lx = padL + 8
      const ly = padT + 10
      VARIANTS.forEach((v) => {
        ctx.strokeStyle = v.color
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(lx, ly)
        ctx.lineTo(lx + 16, ly)
        ctx.stroke()
        ctx.fillStyle = '#ccc'
        ctx.textAlign = 'left'
        ctx.fillText(v.label, lx + 20, ly + 3)
        lx += ctx.measureText(v.label).width + 40
      })
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [smoothed, stds])

  const metrics = VARIANTS.map((v) => ({
    key: v.key,
    label: v.label,
    color: v.color,
    finalReturn: smoothed[v.key][N_EPISODES - 1],
    episodesToConverge: episodesToThreshold(smoothed[v.key], 0.8),
    gradVariance: v.noise * v.noise * 2.6, // synthetic proxy for estimator variance
  }))

  const maxFinal = Math.max(...metrics.map((m) => m.finalReturn), 1)
  const maxEp = Math.max(...metrics.map((m) => m.episodesToConverge), 1)
  const maxVar = Math.max(...metrics.map((m) => m.gradVariance), 0.2)

  return (
    <WidgetFrame
      widgetName="VarianceReductionComparison"
      label="variance reduction — three flavours of REINFORCE"
      right={<span className="font-mono">same seed · bands shrink as baseline → advantage</span>}
      aspect="wide"
    >
      <div className="absolute inset-0 p-4 grid grid-rows-[1.45fr_1fr] gap-3 overflow-hidden">
        <div ref={boxRef} className="min-h-0 relative">
          <canvas ref={canvasRef} className="w-full h-full block" />
        </div>

        <div className="min-h-0 overflow-auto rounded border border-dark-border bg-dark-bg/70 p-3">
          <div className="grid grid-cols-3 gap-5">
            <MetricColumn
              title="final return"
              best="high"
              unit=""
              items={metrics.map((m) => ({
                key: m.key,
                label: m.label,
                color: m.color,
                value: m.finalReturn,
                frac: Math.max(0, m.finalReturn) / maxFinal,
                fmt: (v) => v.toFixed(3),
              }))}
            />
            <MetricColumn
              title="episodes to 80% of optimum"
              best="low"
              unit="ep"
              items={metrics.map((m) => ({
                key: m.key,
                label: m.label,
                color: m.color,
                value: m.episodesToConverge,
                frac: m.episodesToConverge / maxEp,
                fmt: (v) => String(Math.round(v)),
              }))}
            />
            <MetricColumn
              title="gradient estimator variance"
              best="low"
              unit="σ²"
              items={metrics.map((m) => ({
                key: m.key,
                label: m.label,
                color: m.color,
                value: m.gradVariance,
                frac: m.gradVariance / maxVar,
                fmt: (v) => v.toFixed(3),
              }))}
            />
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function MetricColumn({
  title,
  best,
  unit,
  items,
}: {
  title: string
  best: 'low' | 'high'
  unit: string
  items: Array<{
    key: string
    label: string
    color: string
    value: number
    frac: number
    fmt: (v: number) => string
  }>
}) {
  const bestIdx =
    best === 'low'
      ? items.reduce((m, v, i) => (v.value < items[m].value ? i : m), 0)
      : items.reduce((m, v, i) => (v.value > items[m].value ? i : m), 0)
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
        <span>{title}</span>
        <span>{best === 'low' ? '↓ lower better' : '↑ higher better'}</span>
      </div>
      {items.map((m, i) => {
        const isBest = i === bestIdx
        return (
          <div key={m.key} className="flex items-center gap-2 font-mono text-[10.5px]">
            <span
              className={cn('w-32 truncate', isBest ? 'text-dark-text-primary' : 'text-dark-text-secondary')}
              style={{ color: isBest ? m.color : undefined }}
            >
              {m.label}
            </span>
            <div className="flex-1 h-3 bg-dark-surface-elevated/40 rounded-sm overflow-hidden">
              <div
                className="h-full"
                style={{
                  width: `${Math.min(100, m.frac * 100)}%`,
                  backgroundColor: m.color,
                  opacity: isBest ? 0.9 : 0.55,
                }}
              />
            </div>
            <span
              className={cn('w-16 text-right tabular-nums', isBest ? 'font-semibold' : 'text-dark-text-secondary')}
              style={{ color: isBest ? m.color : undefined }}
            >
              {m.fmt(m.value)}
              {unit ? ` ${unit}` : ''}
            </span>
          </div>
        )
      })}
    </div>
  )
}
