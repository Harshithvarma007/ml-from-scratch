'use client'

import { useEffect, useRef, useState } from 'react'
import WidgetFrame, { Button, Readout, Slider } from './WidgetFrame'
import { Play, Pause, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'

// Show a running training simulation. At each step a new batch is drawn from
// a drifting distribution. Three quantities accumulate: batch mean,
// batch variance, and the exponentially-moving running_mean and running_var
// that BatchNorm tracks internally. When we flip to "eval" mode the running
// stats freeze and every incoming sample is normalized with them.

const FEATURES = 4
const BATCH_SIZE = 32

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

// "True" per-feature distribution that slowly drifts — simulates the effect
// of upstream layers changing as training progresses.
function trueStats(step: number): { mean: number; std: number }[] {
  return Array.from({ length: FEATURES }, (_, f) => ({
    mean: 0.5 + 0.4 * Math.sin(step * 0.01 + f),
    std: 1 + 0.5 * Math.cos(step * 0.007 + f * 0.5),
  }))
}

function sampleBatch(step: number, rng: () => number): number[][] {
  const stats = trueStats(step)
  const batch: number[][] = []
  for (let i = 0; i < BATCH_SIZE; i++) {
    const row: number[] = []
    for (let f = 0; f < FEATURES; f++) {
      row.push(stats[f].mean + gauss(rng) * stats[f].std)
    }
    batch.push(row)
  }
  return batch
}

export default function BatchNormLive() {
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [evalMode, setEvalMode] = useState(false)
  const [momentum, setMomentum] = useState(0.1)
  const rafRef = useRef<number | null>(null)
  const lastTickRef = useRef(0)
  const rngRef = useRef<() => number>(mulberry32(3))

  const [runningMean, setRunningMean] = useState<number[]>(new Array(FEATURES).fill(0))
  const [runningVar, setRunningVar] = useState<number[]>(new Array(FEATURES).fill(1))

  // History for the plot
  const historyRef = useRef<{
    batchMean: number[][]
    runMean: number[][]
    batchVar: number[][]
    runVar: number[][]
  }>({ batchMean: [[], [], [], []], runMean: [[], [], [], []], batchVar: [[], [], [], []], runVar: [[], [], [], []] })

  const takeStep = () => {
    const batch = sampleBatch(step, rngRef.current)
    const bMean = new Array(FEATURES).fill(0)
    const bVar = new Array(FEATURES).fill(0)
    for (let f = 0; f < FEATURES; f++) {
      let sum = 0
      for (let i = 0; i < BATCH_SIZE; i++) sum += batch[i][f]
      bMean[f] = sum / BATCH_SIZE
      let v = 0
      for (let i = 0; i < BATCH_SIZE; i++) v += (batch[i][f] - bMean[f]) ** 2
      bVar[f] = v / BATCH_SIZE
    }

    if (!evalMode) {
      // Exponential running stats — the BN training-mode update
      const newRunMean = runningMean.map((m, f) => (1 - momentum) * m + momentum * bMean[f])
      const newRunVar = runningVar.map((v, f) => (1 - momentum) * v + momentum * bVar[f])
      setRunningMean(newRunMean)
      setRunningVar(newRunVar)
      // Record in history
      for (let f = 0; f < FEATURES; f++) {
        historyRef.current.batchMean[f].push(bMean[f])
        historyRef.current.runMean[f].push(newRunMean[f])
        historyRef.current.batchVar[f].push(bVar[f])
        historyRef.current.runVar[f].push(newRunVar[f])
      }
    } else {
      // Eval mode — running stats unchanged, but still record for the plot
      for (let f = 0; f < FEATURES; f++) {
        historyRef.current.batchMean[f].push(bMean[f])
        historyRef.current.runMean[f].push(runningMean[f])
        historyRef.current.batchVar[f].push(bVar[f])
        historyRef.current.runVar[f].push(runningVar[f])
      }
    }

    setStep((s) => s + 1)
  }

  // Animation loop
  useEffect(() => {
    if (!playing) return
    const tick = (t: number) => {
      if (t - lastTickRef.current > 90) {
        lastTickRef.current = t
        takeStep()
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, evalMode, momentum, runningMean, runningVar, step])

  const reset = () => {
    setPlaying(false)
    setStep(0)
    setRunningMean(new Array(FEATURES).fill(0))
    setRunningVar(new Array(FEATURES).fill(1))
    historyRef.current = { batchMean: [[], [], [], []], runMean: [[], [], [], []], batchVar: [[], [], [], []], runVar: [[], [], [], []] }
    rngRef.current = mulberry32(3)
  }

  return (
    <WidgetFrame
      widgetName="BatchNormLive"
      label="batch norm — training-mode statistics and the running mean"
      right={
        <>
          <span className="font-mono">batch {BATCH_SIZE} · {FEATURES} features</span>
          <span className="text-dark-text-disabled">·</span>
          <span
            className={cn('font-mono', evalMode ? 'text-term-amber' : 'text-term-green')}
          >
            {evalMode ? 'eval (frozen)' : 'train (updating)'}
          </span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Button onClick={() => setPlaying((p) => !p)} variant="primary">
            {playing ? <><Pause className="w-3 h-3 inline -mt-px mr-1" /> pause</> : <><Play className="w-3 h-3 inline -mt-px mr-1" /> play</>}
          </Button>
          <Button onClick={reset}>
            <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
          </Button>
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono text-dark-text-disabled uppercase tracking-wider">
              mode
            </span>
            {[
              { v: false, l: 'train' },
              { v: true, l: 'eval' },
            ].map((o) => (
              <button
                key={o.l}
                onClick={() => setEvalMode(o.v)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono uppercase transition-all',
                  evalMode === o.v
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
                )}
              >
                {o.l}
              </button>
            ))}
          </div>
          <Slider
            label="momentum"
            value={momentum}
            min={0.01}
            max={0.5}
            step={0.01}
            onChange={setMomentum}
            accent="accent-term-purple"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="step" value={String(step)} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 grid grid-cols-2 gap-4 overflow-hidden">
        <PlotPanel
          title="mean per feature (over time)"
          subtitle="dashed = current batch · solid = running average"
          batch={historyRef.current.batchMean}
          running={historyRef.current.runMean}
          step={step}
          yRange={[-1.5, 1.5]}
        />
        <PlotPanel
          title="variance per feature (over time)"
          subtitle="dashed = current batch · solid = running average"
          batch={historyRef.current.batchVar}
          running={historyRef.current.runVar}
          step={step}
          yRange={[0, 3]}
        />
      </div>
    </WidgetFrame>
  )
}

function PlotPanel({
  title,
  subtitle,
  batch,
  running,
  step,
  yRange,
}: {
  title: string
  subtitle: string
  batch: number[][]
  running: number[][]
  step: number
  yRange: [number, number]
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)
  const colors = ['#67e8f9', '#fbbf24', '#f472b6', '#a78bfa']

  useEffect(() => {
    const canvas = canvasRef.current
    const box = boxRef.current
    if (!canvas || !box) return
    const dpr = window.devicePixelRatio || 1
    const w = box.clientWidth
    const h = box.clientHeight
    canvas.width = w * dpr
    canvas.height = h * dpr
    canvas.style.width = `${w}px`
    canvas.style.height = `${h}px`
    const ctx = canvas.getContext('2d')!
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, w, h)

    const padL = 36
    const padR = 12
    const padT = 18
    const padB = 26
    const plotW = w - padL - padR
    const plotH = h - padT - padB

    const maxStep = Math.max(step, 100)
    const windowStart = Math.max(0, maxStep - 200)
    const toSx = (s: number) =>
      padL + ((s - windowStart) / Math.max(1, maxStep - windowStart)) * plotW
    const toSy = (v: number) =>
      padT + plotH - ((v - yRange[0]) / (yRange[1] - yRange[0])) * plotH

    ctx.strokeStyle = '#1e1e1e'
    ctx.lineWidth = 1
    ctx.font = '10px "JetBrains Mono", monospace'
    ctx.fillStyle = '#555'
    ctx.textAlign = 'right'
    const ticks = [yRange[0], (yRange[0] + yRange[1]) / 2, yRange[1]]
    ticks.forEach((v) => {
      const sy = toSy(v)
      ctx.beginPath()
      ctx.moveTo(padL, sy)
      ctx.lineTo(padL + plotW, sy)
      ctx.stroke()
      ctx.fillText(v.toFixed(1), padL - 4, sy + 3)
    })

    for (let f = 0; f < 4; f++) {
      const color = colors[f]
      const bSeries = batch[f].slice(windowStart)
      const rSeries = running[f].slice(windowStart)

      // Batch (dashed, faded)
      ctx.strokeStyle = color
      ctx.globalAlpha = 0.35
      ctx.setLineDash([2, 3])
      ctx.lineWidth = 1
      ctx.beginPath()
      bSeries.forEach((v, i) => {
        const sx = toSx(windowStart + i)
        const sy = toSy(v)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.stroke()
      ctx.setLineDash([])

      // Running (solid)
      ctx.globalAlpha = 1
      ctx.lineWidth = 2
      ctx.beginPath()
      rSeries.forEach((v, i) => {
        const sx = toSx(windowStart + i)
        const sy = toSy(v)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.stroke()
    }
  }, [batch, running, step, yRange])

  return (
    <div className="border border-dark-border rounded bg-dark-bg flex flex-col overflow-hidden">
      <div className="px-3 py-1.5 border-b border-dark-border bg-dark-surface-elevated/40 flex items-baseline justify-between">
        <span className="text-[11px] font-mono uppercase tracking-wider">{title}</span>
        <span className="text-[10px] font-mono text-dark-text-disabled">{subtitle}</span>
      </div>
      <div ref={boxRef} className="flex-1 min-h-0 relative">
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>
    </div>
  )
}
