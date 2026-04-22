'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Readout, Button } from './WidgetFrame'
import { Play, Pause, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'

// Three training runs side by side on the same loss landscape:
// full-batch GD (smooth), SGD (batch=1, jagged), minibatch (batch=32, balanced).
// All three see the same data in the same order. The reader watches the
// curves fight it out in real time.

const N = 256
const MAX_STEP = 300

function mulberry32(seed: number) {
  let t = seed >>> 0
  return () => {
    t += 0x6d2b79f5
    let x = Math.imul(t ^ (t >>> 15), t | 1)
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61)
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296
  }
}

function gauss(rng: () => number): number {
  const u = Math.max(rng(), 1e-9)
  const v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

// Toy quadratic bowl with noise. Loss for weight w on dataset is
// (1/|S|) Σ (xᵢ · w − yᵢ)²  on a mini-batch S.
function makeData(): { xs: number[]; ys: number[] } {
  const rng = mulberry32(11)
  const xs: number[] = []
  const ys: number[] = []
  const trueW = 0.7
  for (let i = 0; i < N; i++) {
    const x = gauss(rng) * 1.2
    xs.push(x)
    ys.push(trueW * x + gauss(rng) * 0.4)
  }
  return { xs, ys }
}

function mseLoss(w: number, xs: number[], ys: number[]): number {
  let s = 0
  for (let i = 0; i < xs.length; i++) s += (xs[i] * w - ys[i]) ** 2
  return s / xs.length
}

function grad(w: number, xs: number[], ys: number[]): number {
  let g = 0
  for (let i = 0; i < xs.length; i++) g += 2 * xs[i] * (xs[i] * w - ys[i])
  return g / xs.length
}

interface Regime {
  name: string
  batchSize: number
  lr: number
  color: string
}

const REGIMES: Regime[] = [
  { name: 'full batch', batchSize: N, lr: 0.2, color: '#a78bfa' },
  { name: 'minibatch (32)', batchSize: 32, lr: 0.2, color: '#fbbf24' },
  { name: 'SGD (batch=1)', batchSize: 1, lr: 0.1, color: '#f472b6' },
]

interface RunState {
  w: number
  history: number[]
  pointer: number
}

export default function LossCurveSmoothness() {
  const { xs, ys } = useMemo(() => makeData(), [])
  const [playing, setPlaying] = useState(false)
  const [step, setStep] = useState(0)
  const runsRef = useRef<RunState[]>(
    REGIMES.map(() => ({ w: -1, history: [mseLoss(-1, xs, ys)], pointer: 0 })),
  )
  const [, setTick] = useState(0)
  const rafRef = useRef<number | null>(null)
  const lastTickRef = useRef(0)

  const reset = () => {
    setPlaying(false)
    setStep(0)
    runsRef.current = REGIMES.map(() => ({
      w: -1,
      history: [mseLoss(-1, xs, ys)],
      pointer: 0,
    }))
    setTick((t) => t + 1)
  }

  useEffect(() => {
    if (!playing) return
    const tick = (t: number) => {
      if (t - lastTickRef.current > 50) {
        lastTickRef.current = t
        setStep((s) => {
          if (s >= MAX_STEP) {
            setPlaying(false)
            return s
          }
          // Step each regime once
          runsRef.current = runsRef.current.map((run, i) => {
            const regime = REGIMES[i]
            const start = run.pointer
            const idxs: number[] = []
            for (let k = 0; k < regime.batchSize; k++) {
              idxs.push((start + k) % N)
            }
            const bxs = idxs.map((ix) => xs[ix])
            const bys = idxs.map((ix) => ys[ix])
            const g = grad(run.w, bxs, bys)
            const nw = run.w - regime.lr * g
            const newLoss = mseLoss(nw, xs, ys)
            return {
              w: nw,
              history: [...run.history, newLoss],
              pointer: (start + regime.batchSize) % N,
            }
          })
          setTick((ti) => ti + 1)
          return s + 1
        })
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, xs, ys])

  // Canvas plot
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

      const padL = 48
      const padR = 12
      const padT = 18
      const padB = 28
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const allVals: number[] = []
      runsRef.current.forEach((r) => allVals.push(...r.history))
      const yMin = 0
      const yMax = Math.max(1.5, Math.max(...allVals.slice(0, 5)))

      const toSx = (s: number) => padL + (s / MAX_STEP) * plotW
      const toSy = (v: number) =>
        padT + plotH - (Math.min(v, yMax) / (yMax - yMin)) * plotH

      // Axes
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      ;[yMin, yMax / 2, yMax].forEach((v) => {
        const sy = toSy(v)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(v.toFixed(2), padL - 6, sy + 3)
      })
      ctx.textAlign = 'center'
      ;[0, MAX_STEP / 2, MAX_STEP].forEach((s) =>
        ctx.fillText(String(s), toSx(s), padT + plotH + 14),
      )

      // Each run's curve
      runsRef.current.forEach((run, i) => {
        ctx.strokeStyle = REGIMES[i].color
        ctx.lineWidth = 2
        ctx.globalAlpha = 0.9
        ctx.beginPath()
        run.history.forEach((v, s) => {
          const sx = toSx(s)
          const sy = toSy(v)
          if (s === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        ctx.stroke()
        ctx.globalAlpha = 1
      })

      // Legend
      let lx = padL + 8
      const ly = padT + 14
      REGIMES.forEach((r) => {
        ctx.fillStyle = r.color
        ctx.fillRect(lx, ly - 6, 10, 2)
        ctx.fillStyle = '#ccc'
        ctx.textAlign = 'left'
        ctx.fillText(r.name, lx + 14, ly)
        lx += 14 + ctx.measureText(r.name).width + 18
      })
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  })

  return (
    <WidgetFrame
      widgetName="LossCurveSmoothness"
      label="loss curve smoothness — three batch sizes on the same problem"
      right={
        <>
          <span className="font-mono">same data · same LR · different batch</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Button onClick={() => setPlaying((p) => !p)} variant="primary">
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
          <Button onClick={reset}>
            <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
          </Button>
          <span className="text-[11px] font-mono text-dark-text-muted">
            step {step} / {MAX_STEP}
          </span>
          <div className="flex items-center gap-4 ml-auto">
            {REGIMES.map((r, i) => {
              const run = runsRef.current[i]
              const lastLoss = run.history[run.history.length - 1]
              return (
                <Readout
                  key={r.name}
                  label={r.name}
                  value={lastLoss.toFixed(3)}
                  accent={cn({
                    'text-term-purple': r.color === '#a78bfa',
                    'text-term-amber': r.color === '#fbbf24',
                    'text-term-pink': r.color === '#f472b6',
                  })}
                />
              )
            })}
          </div>
        </div>
      }
    >
      <div ref={boxRef} className="absolute inset-0">
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>
      <div className="absolute bottom-2 left-4 right-4 text-[10.5px] font-mono text-dark-text-disabled pointer-events-none">
        full-batch converges smoothly but needs N forward passes per update · SGD is noisy but generalises · minibatch is the compromise every practitioner uses
      </div>
    </WidgetFrame>
  )
}
