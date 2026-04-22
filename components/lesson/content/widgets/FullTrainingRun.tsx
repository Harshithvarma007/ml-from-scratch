'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Button, Readout } from './WidgetFrame'
import { Play, Pause, RotateCcw, StepForward } from 'lucide-react'
import { cn } from '@/lib/utils'

// A complete training run visualised as a long stream of batches. The underlying
// task is a 1-D linear regression on 256 synthetic points. Every batch we do the
// full 5-line loop (zero_grad → forward → loss → backward → step). The curve is
// the loss at each batch; the dashed line is the per-epoch average.

const DATASET_SIZE = 256
const EPOCHS = 8
const LR = 0.08
// Ground-truth line the synthetic data was drawn from.
const TRUE_W = 1.7
const TRUE_B = -0.4
// Spread of noise on y.
const NOISE_STD = 0.6

interface Point {
  x: number
  y: number
}

// Deterministic pseudo-random (mulberry32) so the dataset is the same every reset.
function mulberry32(seed: number) {
  let t = seed >>> 0
  return () => {
    t = (t + 0x6d2b79f5) >>> 0
    let r = t
    r = Math.imul(r ^ (r >>> 15), r | 1)
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61)
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296
  }
}
function gaussian(rand: () => number): number {
  // Box–Muller, good enough.
  const u = Math.max(1e-9, rand())
  const v = rand()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function makeDataset(seed = 7): Point[] {
  const rand = mulberry32(seed)
  const out: Point[] = []
  for (let i = 0; i < DATASET_SIZE; i++) {
    const x = -2.5 + 5 * rand()
    const y = TRUE_W * x + TRUE_B + NOISE_STD * gaussian(rand)
    out.push({ x, y })
  }
  return out
}

// MSE on a single mini-batch; returns loss and gradients.
function batchStep(
  w: number,
  b: number,
  batch: Point[]
): { loss: number; gw: number; gb: number } {
  let loss = 0
  let gw = 0
  let gb = 0
  const N = batch.length
  for (const { x, y } of batch) {
    const yhat = w * x + b
    const r = yhat - y
    loss += r * r
    gw += 2 * r * x
    gb += 2 * r
  }
  return { loss: loss / N, gw: gw / N, gb: gb / N }
}

// Simulate an entire run ahead-of-time so scrubbing is cheap and the animation
// just walks an array. Returns one entry per batch.
interface BatchRecord {
  epoch: number
  batchInEpoch: number
  globalStep: number
  batchLoss: number
  w: number
  b: number
}

function runTraining(
  data: Point[],
  batchSize: number,
  shuffle: boolean,
  seed: number
): { records: BatchRecord[]; batchesPerEpoch: number; epochMean: number[] } {
  const rand = mulberry32(seed)
  let w = -1.2
  let b = 0.8
  const records: BatchRecord[] = []
  const batchesPerEpoch = Math.ceil(data.length / batchSize)
  const epochMean: number[] = []
  const indices = data.map((_, i) => i)

  let gStep = 0
  for (let e = 0; e < EPOCHS; e++) {
    if (shuffle) {
      // Fisher–Yates with our deterministic rand.
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1))
        ;[indices[i], indices[j]] = [indices[j], indices[i]]
      }
    }

    let epochSum = 0
    for (let bi = 0; bi < batchesPerEpoch; bi++) {
      const start = bi * batchSize
      const end = Math.min(start + batchSize, data.length)
      const batch: Point[] = []
      for (let k = start; k < end; k++) batch.push(data[indices[k]])

      const { loss, gw, gb } = batchStep(w, b, batch)
      w -= LR * gw
      b -= LR * gb
      records.push({
        epoch: e,
        batchInEpoch: bi,
        globalStep: gStep,
        batchLoss: loss,
        w,
        b,
      })
      epochSum += loss
      gStep++
    }
    epochMean.push(epochSum / batchesPerEpoch)
  }

  return { records, batchesPerEpoch, epochMean }
}

const BATCH_SIZES = [1, 8, 32, 128]

export default function FullTrainingRun() {
  const data = useMemo(() => makeDataset(7), [])
  const [batchSize, setBatchSize] = useState(32)
  const [shuffle, setShuffle] = useState(true)
  const [seed, setSeed] = useState(11)

  const { records, batchesPerEpoch, epochMean } = useMemo(
    () => runTraining(data, batchSize, shuffle, seed),
    [data, batchSize, shuffle, seed]
  )

  const totalSteps = records.length
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const rafRef = useRef<number | null>(null)
  const lastTickRef = useRef(0)

  // Reset position whenever the run changes.
  useEffect(() => {
    setStep(0)
    setPlaying(false)
  }, [batchSize, shuffle, seed])

  // Playback.
  useEffect(() => {
    if (!playing) return
    const tickInterval = batchSize <= 1 ? 12 : batchSize <= 8 ? 25 : batchSize <= 32 ? 45 : 75
    const tick = (t: number) => {
      if (t - lastTickRef.current > tickInterval) {
        lastTickRef.current = t
        setStep((s) => {
          if (s >= totalSteps - 1) {
            setPlaying(false)
            return totalSteps - 1
          }
          return s + 1
        })
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [playing, totalSteps, batchSize])

  const cur = records[Math.min(step, totalSteps - 1)]

  // Epoch-so-far mean: average the batch losses in the current epoch up to now.
  const epochSoFarMean = useMemo(() => {
    if (!cur) return 0
    let s = 0
    let n = 0
    for (let i = 0; i <= step; i++) {
      if (records[i].epoch === cur.epoch) {
        s += records[i].batchLoss
        n++
      }
    }
    return n === 0 ? 0 : s / n
  }, [step, cur, records])

  // ─────── Main loss-curve canvas ───────
  const curveCanvas = useRef<HTMLCanvasElement | null>(null)
  const curveBox = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    const canvas = curveCanvas.current
    const box = curveBox.current
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

      const padL = 44
      const padR = 12
      const padT = 20
      const padB = 26
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      // Log-ish y-axis helps when loss ranges from ~5 down to ~0.4.
      const maxLoss = Math.max(...records.map((r) => r.batchLoss))
      const minLoss = Math.min(...records.map((r) => r.batchLoss), 0)
      const yTop = Math.max(maxLoss * 1.05, 0.1)
      const yBot = Math.max(0, minLoss * 0.95)

      const toSx = (gs: number) => padL + (gs / Math.max(1, totalSteps - 1)) * plotW
      const toSy = (L: number) => padT + plotH - ((L - yBot) / (yTop - yBot)) * plotH

      // Epoch separators
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#4a4a4a'
      ctx.textAlign = 'center'
      for (let e = 0; e < EPOCHS; e++) {
        const startStep = e * batchesPerEpoch
        const sx = toSx(startStep)
        ctx.beginPath()
        ctx.moveTo(sx, padT)
        ctx.lineTo(sx, padT + plotH)
        ctx.stroke()
        if (e > 0) ctx.fillText(`e${e}`, sx, padT + plotH + 14)
      }
      // Final epoch boundary label
      ctx.fillText(`e${EPOCHS}`, padL + plotW, padT + plotH + 14)

      // Y-axis labels
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      ;[yBot, (yBot + yTop) / 2, yTop].forEach((yv) => {
        ctx.fillText(yv.toFixed(2), padL - 6, toSy(yv) + 3)
      })

      // Full (faded) curve — all batches
      ctx.strokeStyle = '#fbbf24'
      ctx.globalAlpha = 0.2
      ctx.lineWidth = 1
      ctx.beginPath()
      records.forEach((r, i) => {
        const sx = toSx(r.globalStep)
        const sy = toSy(r.batchLoss)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.stroke()

      // Solid portion — up to current step
      ctx.globalAlpha = 1
      ctx.lineWidth = 1.5
      ctx.beginPath()
      for (let i = 0; i <= step; i++) {
        const r = records[i]
        const sx = toSx(r.globalStep)
        const sy = toSy(r.batchLoss)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()

      // Per-epoch mean — dashed cyan
      ctx.strokeStyle = '#67e8f9'
      ctx.setLineDash([5, 4])
      ctx.lineWidth = 1.5
      ctx.beginPath()
      for (let e = 0; e < EPOCHS; e++) {
        const sxStart = toSx(e * batchesPerEpoch)
        const sxEnd = toSx(Math.min((e + 1) * batchesPerEpoch, totalSteps - 1))
        const sy = toSy(epochMean[e])
        ctx.moveTo(sxStart, sy)
        ctx.lineTo(sxEnd, sy)
      }
      ctx.stroke()
      ctx.setLineDash([])

      // Current-position dot
      if (cur) {
        ctx.fillStyle = '#fbbf24'
        ctx.beginPath()
        ctx.arc(toSx(cur.globalStep), toSy(cur.batchLoss), 4, 0, Math.PI * 2)
        ctx.fill()
      }

      // Legend
      ctx.font = '10.5px "JetBrains Mono", monospace'
      ctx.textAlign = 'left'
      ctx.fillStyle = '#fbbf24'
      ctx.fillText('— batch loss', padL + 4, padT + 12)
      ctx.fillStyle = '#67e8f9'
      ctx.fillText('– – epoch mean', padL + 100, padT + 12)
      ctx.fillStyle = '#4a4a4a'
      ctx.fillText(`global step`, padL + plotW - 72, padT + plotH + 14 - 16)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [records, step, cur, batchesPerEpoch, epochMean, totalSteps])

  // ─────── Mini sparkline of last 100 losses ───────
  const sparkCanvas = useRef<HTMLCanvasElement | null>(null)
  const sparkBox = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    const canvas = sparkCanvas.current
    const box = sparkBox.current
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

      const from = Math.max(0, step - 99)
      const window = records.slice(from, step + 1)
      if (window.length < 2) return
      const maxL = Math.max(...window.map((r) => r.batchLoss))
      const minL = Math.min(...window.map((r) => r.batchLoss))
      const pad = 6
      const plotW = w - pad * 2
      const plotH = h - pad * 2
      const toSx = (i: number) => pad + (i / (window.length - 1)) * plotW
      const toSy = (L: number) => {
        const span = Math.max(1e-6, maxL - minL)
        return pad + plotH - ((L - minL) / span) * plotH
      }

      ctx.strokeStyle = '#f472b6'
      ctx.lineWidth = 1.25
      ctx.beginPath()
      window.forEach((r, i) => {
        const sx = toSx(i)
        const sy = toSy(r.batchLoss)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.stroke()

      // Latest dot
      ctx.fillStyle = '#f472b6'
      ctx.beginPath()
      ctx.arc(toSx(window.length - 1), toSy(window[window.length - 1].batchLoss), 2.5, 0, Math.PI * 2)
      ctx.fill()
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [records, step])

  const togglePlay = () => {
    if (step >= totalSteps - 1) setStep(0)
    setPlaying((p) => !p)
  }
  const reset = () => {
    setStep(0)
    setPlaying(false)
    setSeed((s) => s + 1)
  }
  const stepOnce = () => {
    setPlaying(false)
    setStep((s) => Math.min(s + 1, totalSteps - 1))
  }

  const progressPct = ((step + 1) / totalSteps) * 100

  return (
    <WidgetFrame
      widgetName="FullTrainingRun"
      label="a full training run, batch by batch"
      right={
        <>
          <span className="font-mono">
            {EPOCHS} epochs · {batchesPerEpoch} batches/epoch · lr α = {LR}
          </span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-col gap-2.5">
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2">
              <Button onClick={togglePlay} variant="primary">
                {playing ? (
                  <>
                    <Pause className="w-3 h-3 inline -mt-px mr-1" /> pause
                  </>
                ) : (
                  <>
                    <Play className="w-3 h-3 inline -mt-px mr-1" /> play
                  </>
                )}
              </Button>
              <Button onClick={stepOnce} disabled={step >= totalSteps - 1}>
                <StepForward className="w-3 h-3 inline -mt-px mr-1" /> step
              </Button>
              <Button onClick={reset}>
                <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
              </Button>
            </div>
            <div className="flex items-center gap-1 flex-wrap">
              <span className="text-[11px] font-mono text-dark-text-disabled uppercase tracking-wider mr-1">
                batch
              </span>
              {BATCH_SIZES.map((bs) => (
                <button
                  key={bs}
                  onClick={() => setBatchSize(bs)}
                  className={cn(
                    'px-2.5 py-1 rounded text-[11px] font-mono tracking-wider transition-all',
                    batchSize === bs
                      ? 'bg-dark-accent text-white'
                      : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary hover:border-dark-border-hover'
                  )}
                >
                  {bs}
                </button>
              ))}
            </div>
            <label className="flex items-center gap-2 text-[11px] font-mono text-dark-text-secondary cursor-pointer select-none">
              <input
                type="checkbox"
                checked={shuffle}
                onChange={(e) => setShuffle(e.target.checked)}
                className="accent-term-purple"
              />
              shuffle each epoch
            </label>
            <div className="ml-auto flex items-center gap-4">
              <Readout
                label="epoch"
                value={`${cur ? cur.epoch + 1 : 1}/${EPOCHS}`}
                accent="text-term-cyan"
              />
              <Readout
                label="batch"
                value={`${cur ? cur.batchInEpoch + 1 : 1}/${batchesPerEpoch}`}
                accent="text-term-cyan"
              />
              <Readout
                label="batch loss"
                value={cur ? cur.batchLoss.toFixed(4) : '—'}
                accent="text-term-amber"
              />
              <Readout
                label="epoch mean"
                value={epochSoFarMean.toFixed(4)}
                accent="text-term-cyan"
              />
            </div>
          </div>
          <Slider
            label="step"
            value={step}
            min={0}
            max={totalSteps - 1}
            step={1}
            onChange={(n) => {
              setPlaying(false)
              setStep(n)
            }}
            format={(n) =>
              `${String(n + 1).padStart(4, ' ')} / ${totalSteps}   (${progressPct.toFixed(0)}%)`
            }
            accent="accent-term-amber"
          />
        </div>
      }
    >
      <div className="absolute inset-0 grid grid-rows-[1fr_auto] gap-2 p-2">
        <div className="border border-dark-border rounded-md overflow-hidden bg-dark-bg flex flex-col">
          <div className="px-3 py-1.5 border-b border-dark-border bg-dark-surface-elevated/40 flex items-center justify-between">
            <span className="text-[11px] font-mono uppercase tracking-wider">
              loss per batch (faded = future)
            </span>
            <span className="text-[10px] font-mono text-dark-text-disabled">
              w = {cur ? cur.w.toFixed(3) : '—'} &middot; b = {cur ? cur.b.toFixed(3) : '—'} &middot; target w* = {TRUE_W}, b* = {TRUE_B}
            </span>
          </div>
          <div ref={curveBox} className="flex-1 min-h-0 relative">
            <canvas ref={curveCanvas} className="w-full h-full block" />
          </div>
        </div>
        <div className="border border-dark-border rounded-md overflow-hidden bg-dark-bg grid grid-cols-[auto_1fr] items-center">
          <div className="px-3 py-2 border-r border-dark-border">
            <span className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled block">
              last 100
            </span>
            <span className="text-[11px] font-mono text-term-pink tabular-nums">
              {cur ? cur.batchLoss.toFixed(4) : '—'}
            </span>
          </div>
          <div ref={sparkBox} className="relative h-10">
            <canvas ref={sparkCanvas} className="w-full h-full block" />
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
