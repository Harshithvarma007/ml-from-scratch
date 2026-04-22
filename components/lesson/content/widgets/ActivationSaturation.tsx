'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Plot of f(z) and f'(z) for tanh & sigmoid. The user drags z with a slider.
// A vertical cursor shows where we are. The derivative value — the ACTUAL
// factor that appears in the Jacobian product — is reported numerically and
// as a horizontal bar along the bottom. Shows how fast the derivative decays
// in the tails and why saturation kills gradients.

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z))
}

function dsigmoid(z: number): number {
  const s = sigmoid(z)
  return s * (1 - s)
}

function dtanh(z: number): number {
  const t = Math.tanh(z)
  return 1 - t * t
}

export default function ActivationSaturation() {
  const [z, setZ] = useState(0)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const tanhV = Math.tanh(z)
  const sigV = sigmoid(z)
  const dTanhV = dtanh(z)
  const dSigV = dsigmoid(z)

  // Derived: with |derivative| per step = dTanhV, and σ(W) = 1, the product
  // over T steps is dTanhV^T. Show what T is enough to drop to 1e-6.
  const stepsTo1e6Tanh =
    dTanhV <= 0 ? Infinity : Math.ceil(Math.log(1e-6) / Math.log(Math.max(dTanhV, 1e-10)))

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

      const padL = 42
      const padR = 16
      const padT = 18
      const padB = 32
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const xMin = -6
      const xMax = 6
      const yMin = -1.1
      const yMax = 1.1
      const toSx = (x: number) => padL + ((x - xMin) / (xMax - xMin)) * plotW
      const toSy = (y: number) => padT + plotH - ((y - yMin) / (yMax - yMin)) * plotH

      // Gridlines
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      ;[-1, -0.5, 0, 0.5, 1].forEach((y) => {
        const sy = toSy(y)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(y.toFixed(1), padL - 6, sy + 3)
      })
      ctx.textAlign = 'center'
      ;[-6, -3, 0, 3, 6].forEach((x) => ctx.fillText(String(x), toSx(x), padT + plotH + 14))
      ctx.fillStyle = '#777'
      ctx.fillText('z (pre-activation)', padL + plotW / 2, padT + plotH + 26)

      // Zero-y line thick
      ctx.strokeStyle = '#2a2a32'
      ctx.lineWidth = 1.2
      ctx.beginPath()
      ctx.moveTo(padL, toSy(0))
      ctx.lineTo(padL + plotW, toSy(0))
      ctx.stroke()

      // tanh curve
      ctx.strokeStyle = '#67e8f9'
      ctx.lineWidth = 2
      ctx.beginPath()
      for (let i = 0; i <= 200; i++) {
        const x = xMin + (i / 200) * (xMax - xMin)
        const y = Math.tanh(x)
        const sx = toSx(x)
        const sy = toSy(y)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()

      // tanh' curve (scaled to [0,1] already)
      ctx.strokeStyle = 'rgba(103, 232, 249, 0.55)'
      ctx.setLineDash([4, 4])
      ctx.lineWidth = 1.6
      ctx.beginPath()
      for (let i = 0; i <= 200; i++) {
        const x = xMin + (i / 200) * (xMax - xMin)
        const y = dtanh(x)
        const sx = toSx(x)
        const sy = toSy(y)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()
      ctx.setLineDash([])

      // sigmoid curve (shown in [-1,1] range-mapped: 2σ-1 so it visually shares the axis)
      ctx.strokeStyle = '#fbbf24'
      ctx.lineWidth = 2
      ctx.beginPath()
      for (let i = 0; i <= 200; i++) {
        const x = xMin + (i / 200) * (xMax - xMin)
        const y = 2 * sigmoid(x) - 1 // visually compressed to the [-1,1] axis
        const sx = toSx(x)
        const sy = toSy(y)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()

      // σ' curve (peak 0.25, shown on same axis)
      ctx.strokeStyle = 'rgba(251, 191, 36, 0.55)'
      ctx.setLineDash([4, 4])
      ctx.lineWidth = 1.6
      ctx.beginPath()
      for (let i = 0; i <= 200; i++) {
        const x = xMin + (i / 200) * (xMax - xMin)
        const y = dsigmoid(x)
        const sx = toSx(x)
        const sy = toSy(y)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()
      ctx.setLineDash([])

      // Cursor
      const sxZ = toSx(z)
      ctx.strokeStyle = 'rgba(255,255,255,0.35)'
      ctx.setLineDash([3, 4])
      ctx.beginPath()
      ctx.moveTo(sxZ, padT)
      ctx.lineTo(sxZ, padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])

      // Dots
      const drawDot = (x: number, y: number, color: string) => {
        ctx.fillStyle = color
        ctx.beginPath()
        ctx.arc(toSx(x), toSy(y), 4, 0, Math.PI * 2)
        ctx.fill()
      }
      drawDot(z, tanhV, '#67e8f9')
      drawDot(z, 2 * sigV - 1, '#fbbf24')
      drawDot(z, dTanhV, '#67e8f9')
      drawDot(z, dSigV, '#fbbf24')

      // Legend
      ctx.font = '10px "JetBrains Mono", monospace'
      let lx = padL + 10
      const ly = padT + 14
      const lItems: { label: string; color: string; dashed?: boolean }[] = [
        { label: 'tanh(z)', color: '#67e8f9' },
        { label: "tanh'(z)", color: '#67e8f9', dashed: true },
        { label: 'σ(z) (rescaled)', color: '#fbbf24' },
        { label: "σ'(z)", color: '#fbbf24', dashed: true },
      ]
      lItems.forEach(({ label, color, dashed }) => {
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        if (dashed) ctx.setLineDash([3, 3])
        ctx.beginPath()
        ctx.moveTo(lx, ly - 4)
        ctx.lineTo(lx + 14, ly - 4)
        ctx.stroke()
        ctx.setLineDash([])
        ctx.fillStyle = '#ccc'
        ctx.textAlign = 'left'
        ctx.fillText(label, lx + 18, ly)
        lx += 18 + ctx.measureText(label).width + 18
      })
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [z, tanhV, sigV, dTanhV, dSigV])

  const saturated = Math.abs(z) > 3

  return (
    <WidgetFrame
      widgetName="ActivationSaturation"
      label="saturation — tanh & sigmoid and their derivatives"
      right={<span className="font-mono">drag z · watch the derivative budget collapse in the tails</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="z"
            value={z}
            min={-6}
            max={6}
            step={0.05}
            onChange={setZ}
            format={(v) => v.toFixed(2)}
            accent="accent-dark-accent"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="tanh'(z)" value={dTanhV.toExponential(2)} accent={saturated ? 'text-term-rose' : 'text-term-cyan'} />
            <Readout label="σ'(z)" value={dSigV.toExponential(2)} accent={saturated ? 'text-term-rose' : 'text-term-amber'} />
            <Readout
              label="steps to 1e-6"
              value={stepsTo1e6Tanh === Infinity ? '∞' : String(stepsTo1e6Tanh)}
              accent={saturated ? 'text-term-rose' : 'text-dark-text-primary'}
            />
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
