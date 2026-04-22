'use client'

import { useEffect, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// A single plot: binary cross-entropy loss as a function of the predicted
// probability of the true class. The asymmetry is the point — confidently wrong
// is much more expensive than not-quite-right. Drag the cursor to feel it.

export default function LogLossCurve() {
  const [p, setP] = useState(0.7)
  const [yTrue, setYTrue] = useState<0 | 1>(1)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const loss =
    yTrue === 1
      ? -Math.log(Math.max(1e-12, p))
      : -Math.log(Math.max(1e-12, 1 - p))

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
      const padR = 16
      const padT = 24
      const padB = 36
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const yMax = 5
      const toSx = (pv: number) => padL + pv * plotW
      const toSy = (Lv: number) => padT + plotH - (Math.min(Lv, yMax) / yMax) * plotH

      // Grid
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.textAlign = 'center'
      ;[0, 0.25, 0.5, 0.75, 1].forEach((pv) => {
        const sx = toSx(pv)
        ctx.beginPath()
        ctx.moveTo(sx, padT)
        ctx.lineTo(sx, padT + plotH)
        ctx.stroke()
        ctx.fillText(pv.toFixed(2), sx, padT + plotH + 14)
      })
      ctx.textAlign = 'right'
      ;[0, 1, 2, 3, 4].forEach((Lv) => {
        const sy = toSy(Lv)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(String(Lv), padL - 6, sy + 3)
      })

      // Two curves — y=1 (solid amber) and y=0 (dashed rose)
      const drawCurve = (which: 0 | 1, color: string, dashed: boolean) => {
        ctx.strokeStyle = color
        ctx.lineWidth = which === yTrue ? 2 : 1.2
        ctx.globalAlpha = which === yTrue ? 1 : 0.35
        if (dashed) ctx.setLineDash([5, 4])
        ctx.beginPath()
        const N = 200
        for (let i = 0; i <= N; i++) {
          const pv = i / N
          const Lv =
            which === 1
              ? -Math.log(Math.max(1e-6, pv))
              : -Math.log(Math.max(1e-6, 1 - pv))
          const sx = toSx(pv)
          const sy = toSy(Math.min(Lv, yMax))
          if (i === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        ctx.stroke()
        ctx.setLineDash([])
        ctx.globalAlpha = 1
      }
      drawCurve(1, '#fbbf24', false)
      drawCurve(0, '#f472b6', true)

      // Cursor
      const cx = toSx(p)
      const cy = toSy(Math.min(loss, yMax))
      ctx.strokeStyle = '#555'
      ctx.setLineDash([2, 3])
      ctx.beginPath()
      ctx.moveTo(cx, padT)
      ctx.lineTo(cx, padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = yTrue === 1 ? '#fbbf24' : '#f472b6'
      ctx.beginPath()
      ctx.arc(cx, cy, 5, 0, Math.PI * 2)
      ctx.fill()

      // Axis labels
      ctx.fillStyle = '#777'
      ctx.textAlign = 'center'
      ctx.fillText('predicted p(y=1)', padL + plotW / 2, padT + plotH + 26)
      ctx.save()
      ctx.translate(12, padT + plotH / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.fillText('loss', 0, 0)
      ctx.restore()

      // Legend
      ctx.textAlign = 'left'
      ctx.fillStyle = '#fbbf24'
      ctx.fillRect(padL + 4, padT + 4, 10, 2)
      ctx.fillText('y=1   −log p', padL + 18, padT + 8)
      ctx.fillStyle = '#f472b6'
      ctx.fillRect(padL + 4, padT + 20, 10, 2)
      ctx.fillText('y=0   −log(1−p)', padL + 18, padT + 24)
    }

    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [p, yTrue, loss])

  const verdict =
    loss < 0.1
      ? 'near perfect'
      : loss < 0.7
        ? 'reasonable'
        : loss < 2
          ? 'bad'
          : 'catastrophic'

  return (
    <WidgetFrame
      widgetName="LogLossCurve"
      label="binary cross-entropy — how confidence translates to loss"
      right={<span className="font-mono">L = −[y·log p + (1−y)·log(1−p)]</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            <span className="text-[11px] font-mono text-dark-text-disabled uppercase tracking-wider mr-1">
              true y
            </span>
            {[0, 1].map((v) => (
              <button
                key={v}
                onClick={() => setYTrue(v as 0 | 1)}
                className={cn(
                  'px-3 py-1 rounded text-[11px] font-mono transition-all',
                  yTrue === v
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary'
                )}
              >
                y = {v}
              </button>
            ))}
          </div>
          <Slider
            label="predicted p"
            value={p}
            min={0.01}
            max={0.99}
            step={0.005}
            onChange={setP}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="loss" value={loss.toFixed(3)} accent="text-term-amber" />
            <Readout label="verdict" value={verdict} />
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
