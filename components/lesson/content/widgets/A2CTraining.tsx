'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Three curves on the same seed: REINFORCE (high variance, slow),
// A2C (advantage baseline, much tighter), and A2C+GAE (lambda-weighted
// advantage, tightest of all). Top panel plots episode return; bottom
// panel plots critic loss — which only exists for A2C variants. Slider
// for λ (GAE) morphs the third curve between 0 (1-step TD) and 1 (Monte
// Carlo).

const N = 500

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

function simReinforce(seed: number): number[] {
  const rng = mulberry32(seed)
  const out: number[] = []
  for (let t = 0; t < N; t++) {
    const mean = 1 - Math.exp(-t / 180)
    const noise = gauss(rng) * 0.38
    out.push(mean + noise)
  }
  return out
}

function simA2C(seed: number): number[] {
  const rng = mulberry32(seed + 1)
  const out: number[] = []
  for (let t = 0; t < N; t++) {
    const mean = 1 - Math.exp(-t / 130)
    const noise = gauss(rng) * 0.18
    out.push(mean + noise)
  }
  return out
}

function simA2CGAE(seed: number, lambda: number): number[] {
  // λ=0 → 1-step TD → lower bias, higher variance than λ=1 MC but higher var than A2C-basic.
  // λ=1 → Monte Carlo → unbiased but back to REINFORCE-like variance.
  // Sweet spot in between. Use a U-shape for variance around λ≈0.6.
  const varScale = 0.08 + (lambda - 0.6) * (lambda - 0.6) * 0.35
  const speedScale = 1 + (1 - Math.abs(lambda - 0.7)) * 0.2
  const rng = mulberry32(seed + Math.round(lambda * 100) + 2)
  const out: number[] = []
  for (let t = 0; t < N; t++) {
    const mean = 1 - Math.exp(-(t * speedScale) / 110)
    const noise = gauss(rng) * varScale
    out.push(mean + noise)
  }
  return out
}

function simCriticLoss(seed: number, variant: 'a2c' | 'gae', lambda: number): number[] {
  const rng = mulberry32(seed + (variant === 'a2c' ? 3 : 4) + Math.round(lambda * 100))
  const out: number[] = []
  for (let t = 0; t < N; t++) {
    const base = Math.exp(-t / 180) * (variant === 'a2c' ? 0.65 : 0.45 + Math.abs(lambda - 0.7) * 0.3)
    const noise = Math.abs(gauss(rng)) * (0.06 + base * 0.15)
    out.push(Math.max(0.01, base + noise))
  }
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

export default function A2CTraining() {
  const [lambda, setLambda] = useState(0.7)
  const seed = 11

  const runs = useMemo(() => {
    const r = simReinforce(seed)
    const a = simA2C(seed)
    const g = simA2CGAE(seed, lambda)
    return {
      reinforce: rollingMean(r, 25),
      a2c: rollingMean(a, 25),
      gae: rollingMean(g, 25),
    }
  }, [seed, lambda])

  const losses = useMemo(() => {
    return {
      a2c: simCriticLoss(seed, 'a2c', 0),
      gae: simCriticLoss(seed, 'gae', lambda),
    }
  }, [seed, lambda])

  const boxRef = useRef<HTMLDivElement | null>(null)
  const returnCanvas = useRef<HTMLCanvasElement | null>(null)
  const lossCanvas = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = returnCanvas.current
    const box = boxRef.current
    if (!canvas || !box) return
    const dpr = window.devicePixelRatio || 1
    const draw = () => {
      const w = box.clientWidth
      const h = Math.max(140, box.clientHeight * 0.58)
      canvas.width = w * dpr
      canvas.height = h * dpr
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      const ctx = canvas.getContext('2d')!
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, w, h)
      const padL = 40, padR = 14, padT = 14, padB = 22
      const plotW = w - padL - padR
      const plotH = h - padT - padB
      const yMin = -0.3, yMax = 1.3
      const toSx = (t: number) => padL + (t / (N - 1)) * plotW
      const toSy = (v: number) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH

      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.strokeStyle = '#1e1e1e'
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      ;[0, 0.5, 1].forEach((y) => {
        const sy = toSy(y)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(w - padR, sy)
        ctx.stroke()
        ctx.fillText(y.toFixed(1), padL - 6, sy + 3)
      })
      ctx.textAlign = 'center'
      ;[0, 125, 250, 375, 499].forEach((t) => ctx.fillText(String(t), toSx(t), padT + plotH + 14))
      ctx.fillStyle = '#888'
      ctx.textAlign = 'left'
      ctx.fillText('episode return (rolling mean)', padL, padT - 2)

      const drawCurve = (arr: number[], color: string, lw = 2) => {
        ctx.strokeStyle = color
        ctx.lineWidth = lw
        ctx.beginPath()
        for (let t = 0; t < N; t++) {
          const sx = toSx(t)
          const sy = toSy(arr[t])
          if (t === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        ctx.stroke()
      }
      drawCurve(runs.reinforce, '#f472b6')
      drawCurve(runs.a2c, '#67e8f9')
      drawCurve(runs.gae, '#4ade80', 2.4)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [runs])

  useEffect(() => {
    const canvas = lossCanvas.current
    const box = boxRef.current
    if (!canvas || !box) return
    const dpr = window.devicePixelRatio || 1
    const draw = () => {
      const w = box.clientWidth
      const h = Math.max(90, box.clientHeight * 0.4)
      canvas.width = w * dpr
      canvas.height = h * dpr
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      const ctx = canvas.getContext('2d')!
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, w, h)
      const padL = 40, padR = 14, padT = 14, padB = 20
      const plotW = w - padL - padR
      const plotH = h - padT - padB
      const yMin = 0, yMax = 1.0
      const toSx = (t: number) => padL + (t / (N - 1)) * plotW
      const toSy = (v: number) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH

      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.strokeStyle = '#1e1e1e'
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      ;[0, 0.5, 1].forEach((y) => {
        const sy = toSy(y)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(w - padR, sy)
        ctx.stroke()
        ctx.fillText(y.toFixed(1), padL - 6, sy + 3)
      })
      ctx.fillStyle = '#888'
      ctx.textAlign = 'left'
      ctx.fillText('critic loss L_V = (target − V)²', padL, padT - 2)

      const drawCurve = (arr: number[], color: string) => {
        ctx.strokeStyle = color
        ctx.lineWidth = 1.6
        ctx.beginPath()
        for (let t = 0; t < N; t++) {
          const sx = toSx(t)
          const sy = toSy(arr[t])
          if (t === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        ctx.stroke()
      }
      drawCurve(losses.a2c, '#67e8f9')
      drawCurve(losses.gae, '#4ade80')
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [losses])

  const finalReinforce = runs.reinforce[N - 1]
  const finalA2C = runs.a2c[N - 1]
  const finalGAE = runs.gae[N - 1]

  return (
    <WidgetFrame
      widgetName="A2CTraining"
      label="A2C vs REINFORCE vs A2C+GAE"
      right={<span className="font-mono">advantage baseline + GAE → tighter, faster curves</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="λ (GAE)"
            value={lambda}
            min={0}
            max={1}
            step={0.01}
            onChange={setLambda}
            format={(v) => v.toFixed(2)}
            accent="accent-term-green"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="REINFORCE" value={finalReinforce.toFixed(3)} accent="text-term-pink" />
            <Readout label="A2C" value={finalA2C.toFixed(3)} accent="text-term-cyan" />
            <Readout label="A2C+GAE" value={finalGAE.toFixed(3)} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <div ref={boxRef} className="w-full h-full flex flex-col gap-1">
          <div className="flex items-center gap-4 text-[10.5px] font-mono">
            <Legend color="#f472b6" label="REINFORCE" />
            <Legend color="#67e8f9" label="A2C" />
            <Legend color="#4ade80" label={`A2C + GAE (λ = ${lambda.toFixed(2)})`} />
          </div>
          <canvas ref={returnCanvas} className="w-full block" />
          <canvas ref={lossCanvas} className="w-full block" />
        </div>
      </div>
    </WidgetFrame>
  )
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={cn('inline-block w-3 h-[2px]')} style={{ backgroundColor: color }} />
      <span className="text-dark-text-secondary">{label}</span>
    </span>
  )
}
