'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// DPO loss: L = -log σ(β · Δ), where Δ = log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l).
// The x-axis is the log-ratio difference Δ (the implicit reward gap). We plot
// L as a function of Δ for a given β, and overlay the gradient magnitude
// ∂L/∂Δ = -β · σ(-β·Δ). A slider sweeps β from 0.01 → 1.0. A low β flattens
// the curve (weak preference); a high β makes the loss step-like (strict).

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

function dpoLoss(delta: number, beta: number): number {
  // -log σ(β·Δ), clamp to avoid log(0)
  const s = sigmoid(beta * delta)
  return -Math.log(Math.max(s, 1e-9))
}

function dpoGrad(delta: number, beta: number): number {
  // d/dΔ of -log σ(β·Δ) = -β · (1 - σ(β·Δ)) = -β · σ(-β·Δ)
  return -beta * sigmoid(-beta * delta)
}

export default function DPOLossLandscape() {
  const [beta, setBeta] = useState(0.1)
  const [cursor, setCursor] = useState(0)

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const L = useMemo(() => dpoLoss(cursor, beta), [cursor, beta])
  const dLdD = useMemo(() => dpoGrad(cursor, beta), [cursor, beta])
  const implicitReward = beta * cursor

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
      const padR = 52
      const padT = 18
      const padB = 28
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const xMin = -10
      const xMax = 10
      const yLMax = 4 // loss axis
      const yGMax = 1 // gradient axis (right)

      const toSx = (x: number) => padL + ((x - xMin) / (xMax - xMin)) * plotW
      const toSyL = (y: number) => padT + plotH - (y / yLMax) * plotH
      const toSyG = (y: number) => padT + plotH - (Math.abs(y) / yGMax) * plotH

      // Gridlines + axes
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.fillStyle = '#555'
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.textAlign = 'right'
      ;[0, 1, 2, 3, 4].forEach((y) => {
        ctx.beginPath()
        ctx.moveTo(padL, toSyL(y))
        ctx.lineTo(padL + plotW, toSyL(y))
        ctx.stroke()
        ctx.fillText(y.toFixed(0), padL - 6, toSyL(y) + 3)
      })
      ctx.textAlign = 'center'
      ;[-10, -5, 0, 5, 10].forEach((x) => ctx.fillText(String(x), toSx(x), padT + plotH + 14))

      // Right axis labels (gradient)
      ctx.textAlign = 'left'
      ;[0, 0.25, 0.5, 0.75, 1].forEach((y) =>
        ctx.fillText(y.toFixed(2), padL + plotW + 6, toSyG(y) + 3),
      )

      // Zero-line
      ctx.strokeStyle = '#2a2a32'
      ctx.lineWidth = 1.2
      ctx.beginPath()
      ctx.moveTo(toSx(0), padT)
      ctx.lineTo(toSx(0), padT + plotH)
      ctx.stroke()
      ctx.fillStyle = '#555'
      ctx.font = '9.5px "JetBrains Mono", monospace'
      ctx.textAlign = 'center'
      ctx.fillText('decision boundary', toSx(0), padT + 10)

      // Region labels
      ctx.fillStyle = 'rgba(248, 113, 113, 0.5)'
      ctx.fillText('y_l > y_w', toSx(-6), padT + plotH - 6)
      ctx.fillStyle = 'rgba(74, 222, 128, 0.5)'
      ctx.fillText('y_w > y_l', toSx(6), padT + plotH - 6)

      // Comparison β reference curves (ghost)
      const ghostBetas = [0.03, 0.3, 1.0]
      ghostBetas.forEach((gb) => {
        if (Math.abs(gb - beta) < 0.01) return
        ctx.strokeStyle = 'rgba(160, 160, 170, 0.25)'
        ctx.lineWidth = 1
        ctx.setLineDash([3, 3])
        ctx.beginPath()
        for (let i = 0; i <= 200; i++) {
          const x = xMin + (i / 200) * (xMax - xMin)
          const y = Math.min(yLMax, dpoLoss(x, gb))
          const sx = toSx(x)
          const sy = toSyL(y)
          if (i === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        ctx.stroke()
        ctx.setLineDash([])
      })

      // Main loss curve (cyan, filled)
      ctx.fillStyle = 'rgba(103, 232, 249, 0.12)'
      ctx.strokeStyle = '#67e8f9'
      ctx.lineWidth = 2
      ctx.beginPath()
      let first = true
      for (let i = 0; i <= 300; i++) {
        const x = xMin + (i / 300) * (xMax - xMin)
        const y = Math.min(yLMax, dpoLoss(x, beta))
        const sx = toSx(x)
        const sy = toSyL(y)
        if (first) {
          ctx.moveTo(sx, sy)
          first = false
        } else ctx.lineTo(sx, sy)
      }
      ctx.stroke()
      ctx.lineTo(toSx(xMax), toSyL(0))
      ctx.lineTo(toSx(xMin), toSyL(0))
      ctx.closePath()
      ctx.fill()

      // Gradient magnitude (amber, dashed)
      ctx.strokeStyle = 'rgba(251, 191, 36, 0.9)'
      ctx.lineWidth = 1.8
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      for (let i = 0; i <= 300; i++) {
        const x = xMin + (i / 300) * (xMax - xMin)
        const y = Math.abs(dpoGrad(x, beta))
        const sx = toSx(x)
        const sy = toSyG(y)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()
      ctx.setLineDash([])

      // Cursor
      const sxC = toSx(cursor)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)'
      ctx.setLineDash([2, 3])
      ctx.beginPath()
      ctx.moveTo(sxC, padT)
      ctx.lineTo(sxC, padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])

      // Cursor dots
      ctx.fillStyle = '#67e8f9'
      ctx.beginPath()
      ctx.arc(sxC, toSyL(Math.min(yLMax, L)), 4, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = '#fbbf24'
      ctx.beginPath()
      ctx.arc(sxC, toSyG(Math.abs(dLdD)), 4, 0, Math.PI * 2)
      ctx.fill()

      // Implicit reward axis marker
      ctx.fillStyle = '#a78bfa'
      ctx.textAlign = 'center'
      ctx.font = '9.5px "JetBrains Mono", monospace'
      const ticks = [-5, 0, 5]
      ticks.forEach((d) => {
        const r = beta * d
        const sx = toSx(d)
        ctx.fillText(`r=${r.toFixed(2)}`, sx, padT + plotH + 25)
      })

      // Legend
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.textAlign = 'left'
      ctx.fillStyle = '#67e8f9'
      ctx.fillText('loss', padL + 10, padT + 14)
      ctx.fillStyle = '#fbbf24'
      ctx.fillText("|∂L/∂Δ|", padL + 50, padT + 14)
      ctx.fillStyle = '#a78bfa'
      ctx.fillText('implicit reward r = β·Δ', padL + plotW - 140, padT + plotH + 25)

      // Axes titles
      ctx.fillStyle = '#888'
      ctx.textAlign = 'center'
      ctx.fillText('Δ = log π_θ(y_w)/π_ref(y_w) − log π_θ(y_l)/π_ref(y_l)', padL + plotW / 2, padT + plotH + 14 + 26 - 12)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [beta, cursor, L, dLdD])

  return (
    <WidgetFrame
      widgetName="DPOLossLandscape"
      label="DPO loss landscape — β controls preference strictness"
      right={<span className="font-mono">L(Δ) = −log σ(β·Δ)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="β"
            value={beta}
            min={0.01}
            max={1}
            step={0.01}
            onChange={setBeta}
            format={(v) => v.toFixed(2)}
            accent="accent-term-cyan"
          />
          <Slider
            label="Δ"
            value={cursor}
            min={-10}
            max={10}
            step={0.1}
            onChange={setCursor}
            format={(v) => v.toFixed(1)}
            accent="accent-term-pink"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="L" value={L.toFixed(3)} accent="text-term-cyan" />
            <Readout label="|∂L/∂Δ|" value={Math.abs(dLdD).toFixed(3)} accent="text-term-amber" />
            <Readout label="r = β·Δ" value={implicitReward.toFixed(3)} accent="text-term-purple" />
          </div>
        </div>
      }
    >
      <div ref={boxRef} className="absolute inset-0 p-2">
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>
    </WidgetFrame>
  )
}
