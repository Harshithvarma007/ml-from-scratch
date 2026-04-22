'use client'

import { useEffect, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// PPO's clipped surrogate: L_CLIP(r) = min(r·A, clip(r, 1-ε, 1+ε)·A). Two
// subplots, one for A > 0 (we want to push r up, but only to 1+ε) and one
// for A < 0 (we want to push r down, but only to 1-ε). The clip corners are
// labelled and the "unclipped" reference is dashed. Slider for ε.

function lClip(r: number, A: number, eps: number): number {
  const clipped = Math.min(Math.max(r, 1 - eps), 1 + eps)
  return A >= 0 ? Math.min(r * A, clipped * A) : Math.max(r * A, clipped * A)
}

export default function ClippedObjective() {
  const [eps, setEps] = useState(0.2)
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

      const gap = 16
      const subW = (w - gap * 3) / 2
      const padL = 34, padR = 8, padT = 24, padB = 30
      const plotH = h - padT - padB
      const plotW = subW - padL - padR

      const xMin = 0.4, xMax = 1.6
      const yMin = -2.5, yMax = 2.5

      const drawSub = (x0: number, A: number, title: string) => {
        const toSx = (r: number) => x0 + padL + ((r - xMin) / (xMax - xMin)) * plotW
        const toSy = (v: number) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH

        // Backdrop
        ctx.fillStyle = '#0a0a0a'
        ctx.fillRect(x0, 6, subW, h - 12)

        // Title
        ctx.font = '10.5px "JetBrains Mono", monospace'
        ctx.fillStyle = A > 0 ? '#4ade80' : '#fb7185'
        ctx.textAlign = 'left'
        ctx.fillText(title, x0 + padL, 20)

        // Grid
        ctx.font = '9.5px "JetBrains Mono", monospace'
        ctx.strokeStyle = '#1e1e1e'
        ctx.fillStyle = '#555'
        ctx.textAlign = 'right'
        ;[-2, -1, 0, 1, 2].forEach((y) => {
          const sy = toSy(y)
          ctx.beginPath()
          ctx.moveTo(x0 + padL, sy)
          ctx.lineTo(x0 + padL + plotW, sy)
          ctx.stroke()
          ctx.fillText(y.toFixed(0), x0 + padL - 4, sy + 3)
        })
        ctx.textAlign = 'center'
        ;[0.5, 0.75, 1, 1.25, 1.5].forEach((x) => {
          ctx.fillText(x.toFixed(2), toSx(x), padT + plotH + 14)
        })
        ctx.fillStyle = '#777'
        ctx.fillText('r = π_new / π_old', x0 + padL + plotW / 2, padT + plotH + 24)

        // Zero axis
        ctx.strokeStyle = '#2a2a32'
        ctx.lineWidth = 1.2
        ctx.beginPath()
        ctx.moveTo(x0 + padL, toSy(0))
        ctx.lineTo(x0 + padL + plotW, toSy(0))
        ctx.stroke()

        // Clip region shading
        ctx.fillStyle = 'rgba(251, 191, 36, 0.08)'
        const sxL = toSx(1 - eps)
        const sxR = toSx(1 + eps)
        ctx.fillRect(sxL, padT, sxR - sxL, plotH)

        // Unclipped reference r · A  (dashed)
        ctx.strokeStyle = 'rgba(136,136,136,0.65)'
        ctx.setLineDash([4, 4])
        ctx.lineWidth = 1.2
        ctx.beginPath()
        for (let i = 0; i <= 200; i++) {
          const r = xMin + (i / 200) * (xMax - xMin)
          const sx = toSx(r)
          const sy = toSy(r * A)
          if (i === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        ctx.stroke()
        ctx.setLineDash([])

        // Clipped objective — bold
        ctx.strokeStyle = A > 0 ? '#4ade80' : '#fb7185'
        ctx.lineWidth = 2.4
        ctx.beginPath()
        for (let i = 0; i <= 400; i++) {
          const r = xMin + (i / 400) * (xMax - xMin)
          const sx = toSx(r)
          const sy = toSy(lClip(r, A, eps))
          if (i === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        ctx.stroke()

        // Clip-corner labels
        ctx.fillStyle = '#fbbf24'
        ctx.font = '9.5px "JetBrains Mono", monospace'
        ctx.textAlign = 'center'
        ctx.fillText('1−ε', sxL, padT - 4)
        ctx.fillText('1+ε', sxR, padT - 4)
        ctx.setLineDash([2, 4])
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.4)'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(sxL, padT)
        ctx.lineTo(sxL, padT + plotH)
        ctx.moveTo(sxR, padT)
        ctx.lineTo(sxR, padT + plotH)
        ctx.stroke()
        ctx.setLineDash([])

        // r=1 marker
        ctx.strokeStyle = 'rgba(167, 139, 250, 0.55)'
        ctx.setLineDash([3, 3])
        ctx.beginPath()
        ctx.moveTo(toSx(1), padT)
        ctx.lineTo(toSx(1), padT + plotH)
        ctx.stroke()
        ctx.setLineDash([])
        ctx.fillStyle = '#a78bfa'
        ctx.textAlign = 'left'
        ctx.fillText('r=1', toSx(1) + 3, padT + plotH + 12)

        // Kink label
        const kinkR = A > 0 ? 1 + eps : 1 - eps
        ctx.fillStyle = A > 0 ? '#4ade80' : '#fb7185'
        ctx.textAlign = 'left'
        ctx.font = '9.5px "JetBrains Mono", monospace'
        ctx.fillText('kink', toSx(kinkR) + 3, toSy(kinkR * A) - 6)
      }

      drawSub(gap, 1.0, 'A > 0  (A = +1)  ·  update pushes r up until 1+ε')
      drawSub(gap + subW + gap, -1.0, 'A < 0  (A = −1)  ·  update pushes r down until 1−ε')
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [eps])

  return (
    <WidgetFrame
      widgetName="ClippedObjective"
      label="PPO clipped objective"
      right={<span className="font-mono">L_CLIP = min(r·A, clip(r, 1−ε, 1+ε)·A)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="ε"
            value={eps}
            min={0.05}
            max={0.4}
            step={0.01}
            onChange={setEps}
            format={(v) => v.toFixed(2)}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="1 − ε" value={(1 - eps).toFixed(2)} accent="text-term-amber" />
            <Readout label="1 + ε" value={(1 + eps).toFixed(2)} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <div ref={boxRef} className="w-full h-full relative">
          <canvas ref={canvasRef} className="w-full h-full block" />
        </div>
      </div>
    </WidgetFrame>
  )
}
