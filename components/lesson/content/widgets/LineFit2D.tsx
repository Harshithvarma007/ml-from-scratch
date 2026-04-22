'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// A scatter plot with a draggable line. Slope + intercept sliders. Residuals
// draw as vertical segments so "how wrong each prediction is" is visual. The
// readout computes MSE from the same slope/intercept the user is dragging.

interface Point {
  x: number
  y: number
}

// Hand-pick a dataset that isn't perfectly linear (so residuals matter).
function makeData(): Point[] {
  const out: Point[] = []
  const rng = mulberry32(7)
  for (let i = 0; i < 20; i++) {
    const x = -4 + (i / 19) * 8
    const y = 1.4 * x + 0.5 + (rng() - 0.5) * 3.2
    out.push({ x, y })
  }
  return out
}

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const X_MIN = -5
const X_MAX = 5
const Y_MIN = -10
const Y_MAX = 10

export default function LineFit2D() {
  const [w, setW] = useState(0.8)
  const [b, setB] = useState(0)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const data = useMemo(() => makeData(), [])
  const mse = useMemo(() => {
    let s = 0
    for (const { x, y } of data) {
      const r = y - (w * x + b)
      s += r * r
    }
    return s / data.length
  }, [w, b, data])

  useEffect(() => {
    const canvas = canvasRef.current
    const box = boxRef.current
    if (!canvas || !box) return
    const dpr = window.devicePixelRatio || 1

    const draw = () => {
      const cw = box.clientWidth
      const ch = box.clientHeight
      canvas.width = cw * dpr
      canvas.height = ch * dpr
      canvas.style.width = `${cw}px`
      canvas.style.height = `${ch}px`
      const ctx = canvas.getContext('2d')!
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, cw, ch)

      const padL = 42
      const padR = 12
      const padT = 16
      const padB = 30
      const plotW = cw - padL - padR
      const plotH = ch - padT - padB
      const toSx = (x: number) => padL + ((x - X_MIN) / (X_MAX - X_MIN)) * plotW
      const toSy = (y: number) => padT + plotH - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * plotH

      // Grid
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.textAlign = 'center'
      ;[-4, -2, 0, 2, 4].forEach((xv) => {
        const sx = toSx(xv)
        ctx.beginPath()
        ctx.moveTo(sx, padT)
        ctx.lineTo(sx, padT + plotH)
        ctx.stroke()
        ctx.fillText(String(xv), sx, padT + plotH + 14)
      })
      ctx.textAlign = 'right'
      ;[-8, -4, 0, 4, 8].forEach((yv) => {
        const sy = toSy(yv)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(String(yv), padL - 6, sy + 3)
      })

      // Residual sticks (draw first, behind everything)
      data.forEach(({ x, y }) => {
        const yhat = w * x + b
        const sx = toSx(x)
        const syData = toSy(y)
        const syHat = toSy(yhat)
        const r = Math.abs(y - yhat)
        ctx.strokeStyle = r > 2 ? '#f87171' : '#fbbf24'
        ctx.globalAlpha = 0.45
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(sx, syData)
        ctx.lineTo(sx, syHat)
        ctx.stroke()
        ctx.globalAlpha = 1
      })

      // The line
      ctx.strokeStyle = '#a78bfa'
      ctx.lineWidth = 2
      ctx.beginPath()
      const yLeft = w * X_MIN + b
      const yRight = w * X_MAX + b
      ctx.moveTo(toSx(X_MIN), toSy(yLeft))
      ctx.lineTo(toSx(X_MAX), toSy(yRight))
      ctx.stroke()

      // Data points
      data.forEach(({ x, y }) => {
        ctx.fillStyle = '#ffffff'
        ctx.beginPath()
        ctx.arc(toSx(x), toSy(y), 3, 0, Math.PI * 2)
        ctx.fill()
      })

      // Axis labels
      ctx.fillStyle = '#777'
      ctx.textAlign = 'center'
      ctx.fillText('x', padL + plotW / 2, ch - 6)
      ctx.save()
      ctx.translate(12, padT + plotH / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.fillText('y', 0, 0)
      ctx.restore()
    }

    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [data, w, b])

  return (
    <WidgetFrame
      widgetName="LineFit2D"
      label="linear regression — fit by eye"
      right={<span className="font-mono">ŷ = w·x + b</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="slope w"
            value={w}
            min={-2}
            max={3}
            step={0.01}
            onChange={setW}
            accent="accent-term-purple"
          />
          <Slider
            label="intercept b"
            value={b}
            min={-4}
            max={4}
            step={0.05}
            onChange={setB}
            accent="accent-dark-accent"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="MSE" value={mse.toFixed(3)} accent="text-term-amber" />
            <Readout label="n" value={String(data.length)} />
          </div>
        </div>
      }
    >
      <div ref={boxRef} className="absolute inset-0">
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>
      <div className="absolute bottom-2 left-4 text-[10.5px] font-mono text-dark-text-disabled pointer-events-none">
        residual sticks turn rose when |y − ŷ| &gt; 2
      </div>
    </WidgetFrame>
  )
}
