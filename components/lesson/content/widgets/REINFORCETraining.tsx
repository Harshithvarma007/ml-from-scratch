'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Button, Readout } from './WidgetFrame'
import { Play, RotateCcw } from 'lucide-react'

// Simulated REINFORCE training on a tiny contextual bandit. We don't run a
// real policy — we procedurally generate a curve that shows what REINFORCE
// runs actually look like: a noisy climb with a high-variance envelope that
// only shrinks slowly. Seed slider changes the run.

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

function simulateRun(seed: number): number[] {
  const rng = mulberry32(seed)
  const R: number[] = []
  // Target return (bandit optimum) = 1.0. Policy starts near 0, slowly climbs.
  // Noise stays relatively large — REINFORCE's signature issue.
  for (let t = 0; t < N_EPISODES; t++) {
    const mean = 1 - Math.exp(-t / 140)
    const noise = gauss(rng) * Math.max(0.12, 0.42 * Math.exp(-t / 400))
    const spike = rng() < 0.01 ? (rng() - 0.5) * 1.8 : 0
    R.push(Math.max(-0.6, Math.min(1.3, mean + noise + spike)))
  }
  return R
}

function rolling(data: number[], w: number): { mean: number[]; std: number[] } {
  const mean = new Array(data.length).fill(0)
  const std = new Array(data.length).fill(0)
  for (let i = 0; i < data.length; i++) {
    const lo = Math.max(0, i - w + 1)
    const slice = data.slice(lo, i + 1)
    const m = slice.reduce((a, v) => a + v, 0) / slice.length
    mean[i] = m
    const s = Math.sqrt(slice.reduce((a, v) => a + (v - m) ** 2, 0) / slice.length)
    std[i] = s
  }
  return { mean, std }
}

export default function REINFORCETraining() {
  const [seed, setSeed] = useState(7)
  const [playing, setPlaying] = useState(false)
  const [cursor, setCursor] = useState(N_EPISODES)

  const run = useMemo(() => simulateRun(seed), [seed])
  const { mean, std } = useMemo(() => rolling(run, 30), [run])

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  // Animate the cursor when "play" is pressed
  useEffect(() => {
    if (!playing) return
    let handle = 0
    const tick = () => {
      setCursor((c) => {
        if (c >= N_EPISODES) {
          setPlaying(false)
          return N_EPISODES
        }
        return c + 4
      })
      handle = window.setTimeout(tick, 16)
    }
    tick()
    return () => window.clearTimeout(handle)
  }, [playing])

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
      const yMin = -0.6
      const yMax = 1.3
      const toSx = (t: number) => padL + (t / (N_EPISODES - 1)) * plotW
      const toSy = (v: number) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH

      // Gridlines
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
      ctx.fillStyle = '#777'
      ctx.fillText('episode', padL + plotW / 2, padT + plotH + 22)

      // Zero line
      ctx.strokeStyle = '#2a2a32'
      ctx.lineWidth = 1.2
      ctx.beginPath()
      ctx.moveTo(padL, toSy(0))
      ctx.lineTo(padL + plotW, toSy(0))
      ctx.stroke()

      // Variance band
      ctx.fillStyle = 'rgba(167, 139, 250, 0.18)'
      ctx.beginPath()
      for (let t = 0; t < Math.min(cursor, N_EPISODES); t++) {
        const sx = toSx(t)
        const sy = toSy(Math.min(yMax, mean[t] + std[t]))
        if (t === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      for (let t = Math.min(cursor, N_EPISODES) - 1; t >= 0; t--) {
        const sx = toSx(t)
        const sy = toSy(Math.max(yMin, mean[t] - std[t]))
        ctx.lineTo(sx, sy)
      }
      ctx.closePath()
      ctx.fill()

      // Raw return trace (thin)
      ctx.strokeStyle = 'rgba(244, 114, 182, 0.5)'
      ctx.lineWidth = 1.2
      ctx.beginPath()
      for (let t = 0; t < Math.min(cursor, N_EPISODES); t++) {
        const sx = toSx(t)
        const sy = toSy(run[t])
        if (t === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()

      // Smoothed mean
      ctx.strokeStyle = '#a78bfa'
      ctx.lineWidth = 2.2
      ctx.beginPath()
      for (let t = 0; t < Math.min(cursor, N_EPISODES); t++) {
        const sx = toSx(t)
        const sy = toSy(mean[t])
        if (t === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()

      // Target line
      ctx.strokeStyle = 'rgba(74, 222, 128, 0.35)'
      ctx.setLineDash([3, 5])
      ctx.beginPath()
      ctx.moveTo(padL, toSy(1))
      ctx.lineTo(w - padR, toSy(1))
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = '#4ade80'
      ctx.textAlign = 'left'
      ctx.fillText('optimum', padL + 4, toSy(1) - 4)

      // Legend
      let lx = padL + 90
      const ly = padT + 8
      ctx.fillStyle = '#ccc'
      ctx.strokeStyle = 'rgba(244, 114, 182, 0.7)'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(lx, ly)
      ctx.lineTo(lx + 18, ly)
      ctx.stroke()
      ctx.fillText('per-episode return', lx + 22, ly + 3)
      lx += 160
      ctx.strokeStyle = '#a78bfa'
      ctx.lineWidth = 2.2
      ctx.beginPath()
      ctx.moveTo(lx, ly)
      ctx.lineTo(lx + 18, ly)
      ctx.stroke()
      ctx.fillText('30-ep rolling mean ± 1σ', lx + 22, ly + 3)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [run, mean, std, cursor])

  const finalMean = mean[N_EPISODES - 1]
  const finalStd = std[N_EPISODES - 1]

  return (
    <WidgetFrame
      widgetName="REINFORCETraining"
      label="REINFORCE training — noisy climb"
      right={<span className="font-mono">simulated bandit · 500 episodes · variance is the feature</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="seed"
            value={seed}
            min={1}
            max={50}
            step={1}
            onChange={(v) => { setSeed(Math.round(v)); setCursor(N_EPISODES) }}
            format={(v) => String(Math.round(v))}
            accent="accent-term-purple"
          />
          <div className="flex items-center gap-1.5">
            <Button onClick={() => { setCursor(0); setPlaying(true) }} variant="primary" disabled={playing}>
              <span className="inline-flex items-center gap-1">
                <Play size={11} /> run 500
              </span>
            </Button>
            <Button onClick={() => { setPlaying(false); setCursor(N_EPISODES) }}>
              <span className="inline-flex items-center gap-1">
                <RotateCcw size={11} /> show all
              </span>
            </Button>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="final mean" value={finalMean.toFixed(3)} accent="text-term-purple" />
            <Readout label="final σ" value={finalStd.toFixed(3)} accent="text-term-pink" />
            <Readout label="episode" value={`${Math.min(cursor, N_EPISODES)} / ${N_EPISODES}`} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <div ref={boxRef} className="w-full h-full relative">
          <canvas ref={canvasRef} className="w-full h-full block" />
        </div>
      </div>
    </WidgetFrame>
  )
}
