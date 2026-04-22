'use client'

import { useEffect, useRef } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'

// Canonical He et al. 2015 plot: plain vs residual net final accuracy vs depth.
// Plain peaks around depth 20 and degrades after (optimization problem, not
// overfitting). Residual keeps improving, plateaus near the data's entropy.

const DEPTHS = [8, 12, 18, 26, 34, 50, 74, 110, 152]

// Synthetic numbers tuned to the shape from the ResNet paper
const PLAIN_ACC = [0.82, 0.85, 0.87, 0.88, 0.87, 0.85, 0.82, 0.78, 0.74]
const RESIDUAL_ACC = [0.83, 0.86, 0.89, 0.91, 0.925, 0.94, 0.945, 0.948, 0.95]

export default function DepthAblation() {
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
      const padT = 24
      const padB = 32
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const dMin = DEPTHS[0]
      const dMax = DEPTHS[DEPTHS.length - 1]
      const toSx = (d: number) =>
        padL + (Math.log(d / dMin) / Math.log(dMax / dMin)) * plotW
      const toSy = (v: number) => padT + plotH - ((v - 0.7) / 0.3) * plotH

      // Grid
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      ;[0.7, 0.8, 0.9, 0.95, 1.0].forEach((v) => {
        const sy = toSy(v)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(`${Math.round(v * 100)}%`, padL - 6, sy + 3)
      })
      ctx.textAlign = 'center'
      DEPTHS.forEach((d) => ctx.fillText(String(d), toSx(d), padT + plotH + 14))
      ctx.fillStyle = '#777'
      ctx.fillText('depth (log)', padL + plotW / 2, padT + plotH + 24)

      // Plain curve
      ctx.strokeStyle = '#f87171'
      ctx.lineWidth = 2
      ctx.beginPath()
      DEPTHS.forEach((d, i) => {
        const sx = toSx(d)
        const sy = toSy(PLAIN_ACC[i])
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.stroke()
      DEPTHS.forEach((d, i) => {
        ctx.fillStyle = '#f87171'
        ctx.beginPath()
        ctx.arc(toSx(d), toSy(PLAIN_ACC[i]), 3, 0, Math.PI * 2)
        ctx.fill()
      })

      // Residual curve
      ctx.strokeStyle = '#4ade80'
      ctx.lineWidth = 2
      ctx.beginPath()
      DEPTHS.forEach((d, i) => {
        const sx = toSx(d)
        const sy = toSy(RESIDUAL_ACC[i])
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.stroke()
      DEPTHS.forEach((d, i) => {
        ctx.fillStyle = '#4ade80'
        ctx.beginPath()
        ctx.arc(toSx(d), toSy(RESIDUAL_ACC[i]), 3, 0, Math.PI * 2)
        ctx.fill()
      })

      // Annotation: plain degrades
      ctx.fillStyle = '#f87171'
      ctx.textAlign = 'left'
      ctx.font = '11px "JetBrains Mono", monospace'
      ctx.fillText('plain nets degrade past 30 layers', toSx(50), toSy(PLAIN_ACC[5]) - 12)
      // Residual improves
      ctx.fillStyle = '#4ade80'
      ctx.fillText('residual nets keep scaling', toSx(34), toSy(RESIDUAL_ACC[4]) - 14)

      // Legend
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#f87171'
      ctx.fillRect(padL + 6, padT + 6, 10, 2)
      ctx.fillStyle = '#ccc'
      ctx.fillText('plain net', padL + 20, padT + 10)
      ctx.fillStyle = '#4ade80'
      ctx.fillRect(padL + 100, padT + 6, 10, 2)
      ctx.fillStyle = '#ccc'
      ctx.fillText('residual net', padL + 114, padT + 10)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [])

  const plainMax = Math.max(...PLAIN_ACC)
  const plain152 = PLAIN_ACC[PLAIN_ACC.length - 1]
  const resMax = Math.max(...RESIDUAL_ACC)

  return (
    <WidgetFrame
      widgetName="DepthAblation"
      label="plain vs residual — the depth paradox resolved"
      right={<span className="font-mono">CIFAR-10 / ImageNet-style · numbers from He et al. 2015</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <span className="text-[11px] font-mono text-dark-text-muted">
            plain peaks at depth 26, then degrades · residual plateaus near data entropy
          </span>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="plain peak" value={`${(plainMax * 100).toFixed(1)}%`} accent="text-term-rose" />
            <Readout label="plain@152" value={`${(plain152 * 100).toFixed(1)}%`} accent="text-term-rose" />
            <Readout label="residual peak" value={`${(resMax * 100).toFixed(1)}%`} accent="text-term-green" />
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
