'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Button, Readout } from './WidgetFrame'
import { Play, RotateCcw } from 'lucide-react'

// Animate gradient descent on the same regression the loss-bowl widget uses.
// Two panels: a scatter plot with the current line sweeping toward optimum,
// and a loss-vs-step curve on the right. Shows "converges monotonically to the
// closed-form answer" in a single visual.

type Dataset = Array<[number, number]>

function makeData(): Dataset {
  return [
    [-1.8, -2.1],
    [-1.2, -1.3],
    [-0.5, -0.4],
    [0.1, 0.6],
    [0.9, 1.7],
    [1.4, 2.2],
    [2.1, 3.5],
    [2.6, 4.1],
  ]
}

function mse(w: number, b: number, data: Dataset): number {
  let s = 0
  for (const [x, y] of data) {
    const r = y - (w * x + b)
    s += r * r
  }
  return s / data.length
}

function closedForm(data: Dataset): [number, number] {
  const n = data.length
  const xbar = data.reduce((s, [x]) => s + x, 0) / n
  const ybar = data.reduce((s, [, y]) => s + y, 0) / n
  const cov = data.reduce((s, [x, y]) => s + (x - xbar) * (y - ybar), 0)
  const varX = data.reduce((s, [x]) => s + (x - xbar) ** 2, 0)
  const w = cov / varX
  return [w, ybar - w * xbar]
}

function simulate(data: Dataset, lr: number, maxSteps: number): Array<{ w: number; b: number; loss: number }> {
  let w = -0.5
  let b = 1.5
  const out: Array<{ w: number; b: number; loss: number }> = [{ w, b, loss: mse(w, b, data) }]
  for (let i = 0; i < maxSteps; i++) {
    let gw = 0
    let gb = 0
    const N = data.length
    for (const [x, y] of data) {
      const r = y - (w * x + b)
      gw += -2 * x * r
      gb += -2 * r
    }
    gw /= N
    gb /= N
    w -= lr * gw
    b -= lr * gb
    out.push({ w, b, loss: mse(w, b, data) })
  }
  return out
}

const STEPS = 60
const X_MIN = -3
const X_MAX = 3
const Y_MIN = -3
const Y_MAX = 5

export default function GDRace() {
  const data = useMemo(() => makeData(), [])
  const [wStar, bStar] = useMemo(() => closedForm(data), [data])
  const optimumLoss = useMemo(() => mse(wStar, bStar, data), [wStar, bStar, data])

  const [lr, setLr] = useState(0.12)
  const traj = useMemo(() => simulate(data, lr, STEPS), [data, lr])
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const rafRef = useRef<number | null>(null)
  const lastTickRef = useRef(0)

  useEffect(() => {
    setStep(0)
    setPlaying(false)
  }, [lr])

  useEffect(() => {
    if (!playing) return
    const tick = (t: number) => {
      if (t - lastTickRef.current > 80) {
        lastTickRef.current = t
        setStep((s) => {
          if (s >= STEPS) {
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

  const cur = traj[Math.min(step, traj.length - 1)]

  // Scatter + line canvas
  const sctCanvas = useRef<HTMLCanvasElement | null>(null)
  const sctBox = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    const canvas = sctCanvas.current
    const box = sctBox.current
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
      const pad = 28
      const plotW = w - pad * 2
      const plotH = h - pad * 2
      const toSx = (x: number) => pad + ((x - X_MIN) / (X_MAX - X_MIN)) * plotW
      const toSy = (y: number) => pad + plotH - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * plotH

      ctx.strokeStyle = '#1e1e1e'
      ctx.beginPath()
      ctx.moveTo(pad, toSy(0))
      ctx.lineTo(pad + plotW, toSy(0))
      ctx.stroke()

      // Closed-form line (faded)
      ctx.strokeStyle = '#a78bfa'
      ctx.globalAlpha = 0.35
      ctx.setLineDash([4, 4])
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(toSx(X_MIN), toSy(wStar * X_MIN + bStar))
      ctx.lineTo(toSx(X_MAX), toSy(wStar * X_MAX + bStar))
      ctx.stroke()
      ctx.setLineDash([])
      ctx.globalAlpha = 1

      // Current GD line
      ctx.strokeStyle = '#fbbf24'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(toSx(X_MIN), toSy(cur.w * X_MIN + cur.b))
      ctx.lineTo(toSx(X_MAX), toSy(cur.w * X_MAX + cur.b))
      ctx.stroke()

      // Data points
      data.forEach(([x, y]) => {
        ctx.fillStyle = '#ffffff'
        ctx.beginPath()
        ctx.arc(toSx(x), toSy(y), 3, 0, Math.PI * 2)
        ctx.fill()
      })

      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#888'
      ctx.textAlign = 'left'
      ctx.fillText(`ŷ = ${cur.w.toFixed(2)}·x + ${cur.b.toFixed(2)}`, pad + 4, pad + 14)
      ctx.fillStyle = '#a78bfa'
      ctx.fillText(`optimum: ${wStar.toFixed(2)}·x + ${bStar.toFixed(2)}`, pad + 4, pad + 28)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [cur, data, wStar, bStar])

  // Loss curve canvas
  const lossCanvas = useRef<HTMLCanvasElement | null>(null)
  const lossBox = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    const canvas = lossCanvas.current
    const box = lossBox.current
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

      const padL = 40
      const padR = 12
      const padT = 18
      const padB = 28
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const maxLoss = Math.max(...traj.map((t) => t.loss))
      const minLoss = Math.min(optimumLoss * 0.95, 0)

      const toSx = (s: number) => padL + (s / STEPS) * plotW
      const toSy = (L: number) =>
        padT + plotH - ((L - minLoss) / (maxLoss - minLoss)) * plotH

      // Optimum line
      ctx.strokeStyle = '#a78bfa'
      ctx.setLineDash([4, 4])
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(padL, toSy(optimumLoss))
      ctx.lineTo(padL + plotW, toSy(optimumLoss))
      ctx.stroke()
      ctx.setLineDash([])

      // Grid
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.textAlign = 'center'
      ;[0, 15, 30, 45, 60].forEach((sv) => {
        const sx = toSx(sv)
        ctx.beginPath()
        ctx.moveTo(sx, padT)
        ctx.lineTo(sx, padT + plotH)
        ctx.stroke()
        ctx.fillText(String(sv), sx, padT + plotH + 14)
      })
      ctx.textAlign = 'right'
      ctx.fillStyle = '#888'
      ctx.fillText(`L* = ${optimumLoss.toFixed(3)}`, padL + plotW - 12, toSy(optimumLoss) - 4)

      // Full curve, faded after step
      ctx.strokeStyle = '#fbbf24'
      ctx.globalAlpha = 0.3
      ctx.lineWidth = 1.2
      ctx.beginPath()
      traj.forEach((t, i) => {
        const sx = toSx(i)
        const sy = toSy(t.loss)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.stroke()

      // Up-to-cursor segment solid
      ctx.globalAlpha = 1
      ctx.lineWidth = 2
      ctx.beginPath()
      traj.slice(0, step + 1).forEach((t, i) => {
        const sx = toSx(i)
        const sy = toSy(t.loss)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.stroke()

      // Cursor dot
      ctx.fillStyle = '#fbbf24'
      ctx.beginPath()
      ctx.arc(toSx(step), toSy(cur.loss), 4, 0, Math.PI * 2)
      ctx.fill()

      // Label
      ctx.fillStyle = '#777'
      ctx.textAlign = 'center'
      ctx.fillText('step', padL + plotW / 2, padT + plotH + 24)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [traj, step, cur, optimumLoss])

  const play = () => {
    if (step >= STEPS) setStep(0)
    setPlaying(true)
  }
  const reset = () => {
    setStep(0)
    setPlaying(false)
  }

  return (
    <WidgetFrame
      widgetName="GDRace"
      label="gradient descent finds what closed form already knows"
      right={
        <>
          <span className="font-mono">
            α · ∇L · step
          </span>
          <span className="text-dark-text-disabled">·</span>
          <span>dashed line = OLS optimum</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="α"
            value={lr}
            min={0.01}
            max={0.4}
            step={0.005}
            onChange={setLr}
            accent="accent-term-purple"
          />
          <Slider
            label="step"
            value={step}
            min={0}
            max={STEPS}
            step={1}
            onChange={setStep}
            format={(n) => String(n).padStart(2, ' ') + '/' + String(STEPS)}
          />
          <div className="flex items-center gap-2">
            <Button onClick={play} variant="primary" disabled={playing}>
              <Play className="w-3 h-3 inline -mt-px mr-1" /> play
            </Button>
            <Button onClick={reset}>
              <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
            </Button>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="loss" value={cur.loss.toFixed(3)} accent="text-term-amber" />
            <Readout
              label="gap to opt"
              value={(cur.loss - optimumLoss).toFixed(3)}
              accent="text-term-purple"
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 grid grid-cols-[1fr_1fr] gap-2 p-2">
        <div className="border border-dark-border rounded-md overflow-hidden bg-dark-bg flex flex-col">
          <div className="px-3 py-1.5 border-b border-dark-border bg-dark-surface-elevated/40">
            <span className="text-[11px] font-mono uppercase tracking-wider">
              data + current line
            </span>
          </div>
          <div ref={sctBox} className="flex-1 min-h-0 relative">
            <canvas ref={sctCanvas} className="w-full h-full block" />
          </div>
        </div>
        <div className="border border-dark-border rounded-md overflow-hidden bg-dark-bg flex flex-col">
          <div className="px-3 py-1.5 border-b border-dark-border bg-dark-surface-elevated/40">
            <span className="text-[11px] font-mono uppercase tracking-wider">
              loss over steps
            </span>
          </div>
          <div ref={lossBox} className="flex-1 min-h-0 relative">
            <canvas ref={lossCanvas} className="w-full h-full block" />
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
