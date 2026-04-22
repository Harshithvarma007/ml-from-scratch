'use client'

import { useEffect, useRef, useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { Play, Pause, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'

// Three runs' train/val curves on the same CIFAR-style task. No aug overfits
// heavily (train → 0, val plateaus low). Aug closes the gap. Aug + label
// smoothing + cosine LR tightens further. Animate across 50 epochs.

const EPOCHS = 50

// Pre-shaped curves. Each run has train_acc and val_acc arrays.
function run(seed: number, finalTrain: number, finalVal: number, overfit: number): { train: number[]; val: number[] } {
  const train: number[] = []
  const val: number[] = []
  for (let e = 0; e < EPOCHS; e++) {
    const t = e / (EPOCHS - 1)
    const ta = 0.1 + (finalTrain - 0.1) * (1 - Math.exp(-3.2 * t))
    // Val follows train up to a point, then plateaus
    const vBase = 0.1 + (finalVal - 0.1) * (1 - Math.exp(-2.4 * t))
    const vDrop = overfit > 0 ? overfit * Math.max(0, t - 0.4) * 0.3 : 0
    const va = Math.max(0.1, vBase - vDrop)
    train.push(ta)
    val.push(va)
  }
  return { train, val }
}

const RUNS = [
  { name: 'no aug', color: '#f87171', ...run(1, 0.999, 0.74, 0.5) },
  { name: '+ augmentation', color: '#fbbf24', ...run(2, 0.95, 0.88, 0.1) },
  { name: '+ label smoothing + cosine LR', color: '#4ade80', ...run(3, 0.92, 0.94, 0.0) },
]

export default function TrainValCurves() {
  const [step, setStep] = useState(EPOCHS)
  const [playing, setPlaying] = useState(false)
  const rafRef = useRef<number | null>(null)
  const lastTickRef = useRef(0)

  useEffect(() => {
    if (!playing) return
    const tick = (t: number) => {
      if (t - lastTickRef.current > 80) {
        lastTickRef.current = t
        setStep((s) => {
          if (s >= EPOCHS) {
            setPlaying(false)
            return s
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
  }, [playing])

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

      const padL = 44
      const padR = 12
      const padT = 18
      const padB = 28
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const toSx = (e: number) => padL + (e / (EPOCHS - 1)) * plotW
      const toSy = (v: number) => padT + plotH - v * plotH

      // Grid
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      ;[0, 0.25, 0.5, 0.75, 1].forEach((v) => {
        const sy = toSy(v)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(`${Math.round(v * 100)}%`, padL - 6, sy + 3)
      })
      ctx.textAlign = 'center'
      ;[0, 25, 49].forEach((e) => ctx.fillText(String(e), toSx(e), padT + plotH + 14))
      ctx.fillStyle = '#777'
      ctx.fillText('epoch', padL + plotW / 2, padT + plotH + 24)

      RUNS.forEach((r) => {
        // Train solid
        ctx.strokeStyle = r.color
        ctx.lineWidth = 2
        ctx.beginPath()
        r.train.slice(0, step).forEach((v, e) => {
          const sx = toSx(e)
          const sy = toSy(v)
          if (e === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        ctx.stroke()
        // Val dashed
        ctx.setLineDash([4, 3])
        ctx.lineWidth = 1.8
        ctx.globalAlpha = 0.8
        ctx.beginPath()
        r.val.slice(0, step).forEach((v, e) => {
          const sx = toSx(e)
          const sy = toSy(v)
          if (e === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        ctx.stroke()
        ctx.setLineDash([])
        ctx.globalAlpha = 1
      })

      // Legend
      let lx = padL + 8
      const ly = padT + 14
      RUNS.forEach((r) => {
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
  }, [step])

  const reset = () => {
    setPlaying(false)
    setStep(0)
  }

  return (
    <WidgetFrame
      widgetName="TrainValCurves"
      label="train vs val — three runs, three regularizers"
      right={<span className="font-mono">solid = train acc · dashed = val acc</span>}
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
            epoch {step} / {EPOCHS}
          </span>
          <div className="flex items-center gap-4 ml-auto">
            {RUNS.map((r, i) => {
              const tv = r.train[Math.min(step, EPOCHS - 1)]
              const vv = r.val[Math.min(step, EPOCHS - 1)]
              return (
                <Readout
                  key={i}
                  label={r.name}
                  value={`${(tv * 100).toFixed(0)}/${(vv * 100).toFixed(0)}%`}
                  accent={cn({
                    'text-term-rose': r.color === '#f87171',
                    'text-term-amber': r.color === '#fbbf24',
                    'text-term-green': r.color === '#4ade80',
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
    </WidgetFrame>
  )
}
