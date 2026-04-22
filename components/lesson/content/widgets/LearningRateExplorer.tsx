'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Button, Readout } from './WidgetFrame'
import { RotateCcw, Play } from 'lucide-react'
import { cn } from '@/lib/utils'

// f(x) = x², f'(x) = 2x. One-dimensional side-view so the reader can SEE the
// step length and overshoot behavior change as α is dragged.

const X0 = 5
const MAX_STEPS = 40

function simulate(lr: number): { xs: number[]; diverged: boolean } {
  const xs: number[] = [X0]
  let x = X0
  let diverged = false
  for (let i = 0; i < MAX_STEPS; i++) {
    const next = x - lr * 2 * x
    if (!isFinite(next) || Math.abs(next) > 1e4) {
      diverged = true
      break
    }
    xs.push(next)
    x = next
  }
  return { xs, diverged }
}

export default function LearningRateExplorer() {
  const [lr, setLr] = useState(0.1)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const rafRef = useRef<number | null>(null)
  const lastTickRef = useRef(0)

  const sim = useMemo(() => simulate(lr), [lr])
  const maxStep = sim.xs.length - 1
  const visible = Math.min(step, maxStep)

  useEffect(() => {
    setStep(0)
    setPlaying(false)
  }, [lr])

  useEffect(() => {
    if (!playing) return
    const tick = (t: number) => {
      if (t - lastTickRef.current > 120) {
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

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return
    const dpr = window.devicePixelRatio || 1

    const draw = () => {
      const w = container.clientWidth
      const h = container.clientHeight
      canvas.width = w * dpr
      canvas.height = h * dpr
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      const ctx = canvas.getContext('2d')!
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, w, h)

      // Coord system: x in [-6, 6], y in [0, 30]
      const xMin = -6
      const xMax = 6
      const yMin = 0
      const yMax = 30

      const toSx = (x: number) => ((x - xMin) / (xMax - xMin)) * w
      const toSy = (y: number) => h - ((y - yMin) / (yMax - yMin)) * (h - 40) - 30

      // Axes
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(0, toSy(0))
      ctx.lineTo(w, toSy(0))
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(toSx(0), 0)
      ctx.lineTo(toSx(0), h)
      ctx.stroke()

      // Grid labels
      ctx.fillStyle = '#555'
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.textAlign = 'center'
      ;[-5, -3, -1, 1, 3, 5].forEach((x) => ctx.fillText(String(x), toSx(x), h - 8))
      ctx.textAlign = 'left'
      ;[10, 20].forEach((y) => ctx.fillText(String(y), 4, toSy(y) - 2))

      // Parabola
      ctx.strokeStyle = '#a78bfa'
      ctx.globalAlpha = 0.6
      ctx.lineWidth = 1.5
      ctx.beginPath()
      for (let sx = 0; sx <= w; sx += 2) {
        const x = xMin + (sx / w) * (xMax - xMin)
        const y = x * x
        const sy = toSy(y)
        if (sx === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()
      ctx.globalAlpha = 1

      // Trail — only up through `visible`
      const xs = sim.xs.slice(0, visible + 1)
      if (xs.length > 1) {
        ctx.strokeStyle = '#fbbf24'
        ctx.globalAlpha = 0.5
        ctx.lineWidth = 1.5
        ctx.beginPath()
        xs.forEach((x, i) => {
          const sx = toSx(x)
          const sy = toSy(x * x)
          if (i === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        ctx.stroke()
        ctx.globalAlpha = 1

        // Dots at each step
        xs.forEach((x, i) => {
          const sx = toSx(x)
          const sy = toSy(x * x)
          ctx.fillStyle = '#fbbf24'
          ctx.globalAlpha = 0.3 + (i / xs.length) * 0.7
          ctx.beginPath()
          ctx.arc(sx, sy, 3, 0, Math.PI * 2)
          ctx.fill()
        })
        ctx.globalAlpha = 1
      }

      // Current marble
      const cur = sim.xs[visible]
      if (cur !== undefined && isFinite(cur) && Math.abs(cur) < 6) {
        const sx = toSx(cur)
        const sy = toSy(cur * cur)
        ctx.shadowColor = '#fbbf24'
        ctx.shadowBlur = 12
        ctx.fillStyle = '#fbbf24'
        ctx.beginPath()
        ctx.arc(sx, sy, 6, 0, Math.PI * 2)
        ctx.fill()
        ctx.shadowBlur = 0
      }

      // Multiplier readout at top
      const mult = 1 - 2 * lr
      const badge = `(1 − 2α) = ${mult.toFixed(2)}`
      const verdict =
        Math.abs(mult) < 1
          ? 'converges'
          : Math.abs(mult) > 1
            ? 'diverges'
            : 'oscillates forever'
      const verdictColor =
        Math.abs(mult) < 1 ? '#4ade80' : Math.abs(mult) > 1 ? '#f87171' : '#fbbf24'

      ctx.font = '11px "JetBrains Mono", monospace'
      ctx.fillStyle = '#888'
      ctx.textAlign = 'right'
      ctx.fillText(badge, w - 12, 18)
      ctx.fillStyle = verdictColor
      ctx.fillText(verdict, w - 12, 34)
    }

    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(container)
    return () => ro.disconnect()
  }, [sim, visible, lr])

  const currentX = sim.xs[visible]
  const currentLoss =
    currentX !== undefined && isFinite(currentX) ? currentX * currentX : NaN

  const reset = () => {
    setStep(0)
    setPlaying(false)
  }
  const play = () => {
    if (step >= maxStep) setStep(0)
    setPlaying(true)
  }

  return (
    <WidgetFrame
      widgetName="LearningRateExplorer"
      label="learning rate — convergence and divergence"
      right={<span>f(x) = x² · side view</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="α"
            value={lr}
            min={0.0}
            max={1.2}
            step={0.01}
            onChange={setLr}
            accent={
              1 - 2 * lr > 0
                ? 'accent-dark-accent'
                : Math.abs(1 - 2 * lr) > 1
                  ? 'accent-term-rose'
                  : 'accent-term-amber'
            }
          />
          <Slider
            label="step"
            value={step}
            min={0}
            max={maxStep}
            step={1}
            onChange={setStep}
            format={(n) => String(n).padStart(2, ' ') + '/' + String(maxStep)}
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
              label="x"
              value={isFinite(currentX) ? currentX.toFixed(3) : '∞'}
            />
            <Readout
              label="f(x)"
              value={isFinite(currentLoss) ? currentLoss.toFixed(3) : '∞'}
              accent={
                sim.diverged && visible === maxStep ? 'text-term-rose' : 'text-term-amber'
              }
            />
          </div>
        </div>
      }
    >
      <div ref={containerRef} className="absolute inset-0">
        <canvas
          ref={canvasRef}
          className={cn(
            'w-full h-full block',
            sim.diverged && visible === maxStep && 'animate-pulse'
          )}
        />
      </div>
    </WidgetFrame>
  )
}
