'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Button, Readout } from './WidgetFrame'
import { Play, RotateCcw } from 'lucide-react'

// A narrow ravine — tiny curvature along x, huge curvature along y. This is
// the canonical case where vanilla GD zig-zags and momentum eats it for lunch.
//
//   f(x, y) = 0.05·x² + 4·y²
//   ∇f = (0.1·x,  8·y)

const START: [number, number] = [-4.5, 1.5]
const MAX_STEPS = 120

function grad(x: number, y: number): [number, number] {
  return [0.1 * x, 8 * y]
}
function loss(x: number, y: number) {
  return 0.05 * x * x + 4 * y * y
}

function simulateVanilla(lr: number): Array<[number, number]> {
  const out: Array<[number, number]> = [START]
  let [x, y] = START
  for (let i = 0; i < MAX_STEPS; i++) {
    const [gx, gy] = grad(x, y)
    x -= lr * gx
    y -= lr * gy
    if (!isFinite(x) || !isFinite(y) || Math.abs(x) > 50 || Math.abs(y) > 50) break
    out.push([x, y])
  }
  return out
}

function simulateMomentum(lr: number, beta: number): Array<[number, number]> {
  const out: Array<[number, number]> = [START]
  let [x, y] = START
  let vx = 0
  let vy = 0
  for (let i = 0; i < MAX_STEPS; i++) {
    const [gx, gy] = grad(x, y)
    vx = beta * vx - lr * gx
    vy = beta * vy - lr * gy
    x += vx
    y += vy
    if (!isFinite(x) || !isFinite(y) || Math.abs(x) > 50 || Math.abs(y) > 50) break
    out.push([x, y])
  }
  return out
}

interface PanelProps {
  title: string
  subtitle: string
  path: Array<[number, number]>
  step: number
  color: string
}

function Panel({ title, subtitle, path, step, color }: PanelProps) {
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

      const xMin = -6
      const xMax = 6
      const yMin = -3
      const yMax = 3

      const toSx = (x: number) => ((x - xMin) / (xMax - xMin)) * w
      const toSy = (y: number) => h - ((y - yMin) / (yMax - yMin)) * h

      // Contour lines — iso-loss curves look like narrow horizontal ellipses.
      const levels = [0.2, 0.8, 2, 4, 8, 16, 32]
      levels.forEach((L) => {
        ctx.beginPath()
        // f = L  ⇒  0.05x² + 4y² = L.  Param: x = √(L/0.05)·cosθ, y = √(L/4)·sinθ.
        const rx = Math.sqrt(L / 0.05)
        const ry = Math.sqrt(L / 4)
        for (let t = 0; t <= Math.PI * 2 + 0.01; t += 0.05) {
          const x = rx * Math.cos(t)
          const y = ry * Math.sin(t)
          const sx = toSx(x)
          const sy = toSy(y)
          if (t === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        ctx.strokeStyle = '#a78bfa'
        ctx.globalAlpha = 0.15
        ctx.lineWidth = 1
        ctx.stroke()
      })
      ctx.globalAlpha = 1

      // Minimum marker
      ctx.fillStyle = '#a78bfa'
      ctx.globalAlpha = 0.6
      ctx.beginPath()
      ctx.arc(toSx(0), toSy(0), 3, 0, Math.PI * 2)
      ctx.fill()
      ctx.globalAlpha = 1

      // Trail
      const visible = path.slice(0, step + 1)
      if (visible.length > 1) {
        ctx.strokeStyle = color
        ctx.globalAlpha = 0.6
        ctx.lineWidth = 1.5
        ctx.beginPath()
        visible.forEach(([x, y], i) => {
          const sx = toSx(x)
          const sy = toSy(y)
          if (i === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        ctx.stroke()
        ctx.globalAlpha = 1

        visible.forEach(([x, y], i) => {
          ctx.fillStyle = color
          ctx.globalAlpha = 0.25 + (i / visible.length) * 0.6
          ctx.beginPath()
          ctx.arc(toSx(x), toSy(y), 2, 0, Math.PI * 2)
          ctx.fill()
        })
        ctx.globalAlpha = 1
      }

      // Current marble
      const cur = path[Math.min(step, path.length - 1)]
      if (cur) {
        ctx.shadowColor = color
        ctx.shadowBlur = 10
        ctx.fillStyle = color
        ctx.beginPath()
        ctx.arc(toSx(cur[0]), toSy(cur[1]), 5, 0, Math.PI * 2)
        ctx.fill()
        ctx.shadowBlur = 0
      }
    }

    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [path, step, color])

  const cur = path[Math.min(step, path.length - 1)]
  const curLoss = cur ? loss(cur[0], cur[1]) : NaN

  return (
    <div className="flex flex-col border border-dark-border rounded-md overflow-hidden bg-dark-bg">
      <div className="flex items-baseline justify-between px-3 py-1.5 border-b border-dark-border bg-dark-surface-elevated/40">
        <div className="flex items-baseline gap-2">
          <span className="text-[11px] font-mono uppercase tracking-wider text-dark-text-primary">
            {title}
          </span>
          <span className="text-[10px] font-mono text-dark-text-disabled">{subtitle}</span>
        </div>
        <span
          className="text-[11px] font-mono tabular-nums"
          style={{ color }}
        >
          loss {isFinite(curLoss) ? curLoss.toFixed(3) : '∞'}
        </span>
      </div>
      <div ref={boxRef} className="relative flex-1 min-h-0">
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>
    </div>
  )
}

export default function MomentumCompare() {
  const [lr, setLr] = useState(0.11)
  const [beta, setBeta] = useState(0.85)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const rafRef = useRef<number | null>(null)
  const lastTickRef = useRef(0)

  const vanilla = useMemo(() => simulateVanilla(lr), [lr])
  const momentum = useMemo(() => simulateMomentum(lr, beta), [lr, beta])
  const maxStep = Math.max(vanilla.length, momentum.length) - 1

  useEffect(() => {
    setStep(0)
    setPlaying(false)
  }, [lr, beta])

  useEffect(() => {
    if (!playing) return
    const tick = (t: number) => {
      if (t - lastTickRef.current > 60) {
        lastTickRef.current = t
        setStep((s) => {
          if (s >= maxStep) {
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
  }, [playing, maxStep])

  const play = () => {
    if (step >= maxStep) setStep(0)
    setPlaying(true)
  }
  const reset = () => {
    setStep(0)
    setPlaying(false)
  }

  const vLoss = loss(...(vanilla[Math.min(step, vanilla.length - 1)] ?? [NaN, NaN]))
  const mLoss = loss(...(momentum[Math.min(step, momentum.length - 1)] ?? [NaN, NaN]))

  return (
    <WidgetFrame
      widgetName="MomentumCompare"
      label="vanilla vs momentum — a narrow ravine"
      right={
        <>
          <span>f(x,y) = 0.05x² + 4y²</span>
          <span className="text-dark-text-disabled">·</span>
          <span>same start · same α</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="α"
            value={lr}
            min={0.02}
            max={0.24}
            step={0.005}
            onChange={setLr}
            accent="accent-dark-accent"
          />
          <Slider
            label="β"
            value={beta}
            min={0.0}
            max={0.98}
            step={0.01}
            onChange={setBeta}
            accent="accent-term-amber"
          />
          <Slider
            label="step"
            value={step}
            min={0}
            max={maxStep}
            step={1}
            onChange={setStep}
            format={(n) => String(n).padStart(3, ' ') + '/' + String(maxStep)}
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
            <Readout
              label="Δ loss"
              value={
                isFinite(vLoss - mLoss)
                  ? (vLoss - mLoss >= 0 ? '+' : '') + (vLoss - mLoss).toFixed(3)
                  : '∞'
              }
              accent="text-term-amber"
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 grid grid-cols-2 gap-2 p-2">
        <Panel
          title="vanilla GD"
          subtitle="x ← x − α∇f"
          path={vanilla}
          step={step}
          color="#60a5fa"
        />
        <Panel
          title="momentum"
          subtitle="v ← βv − α∇f ;  x ← x + v"
          path={momentum}
          step={step}
          color="#fbbf24"
        />
      </div>
    </WidgetFrame>
  )
}
