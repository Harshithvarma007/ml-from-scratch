'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Fixed logits, sweep T. Two panels: bar chart of probabilities on the left,
// entropy-vs-T curve on the right with a moving cursor. Makes the two limits
// visible — one-hot as T→0, uniform as T→∞.

const LOGITS = [2.0, 1.2, 0.6, -0.3, -1.1]
const CLASSES = ['A', 'B', 'C', 'D', 'E']
const T_MIN = 0.05
const T_MAX = 5.0

function softmax(z: number[], T: number): number[] {
  const t = Math.max(1e-6, T)
  const s = z.map((v) => v / t)
  const m = Math.max(...s)
  const e = s.map((v) => Math.exp(v - m))
  const sum = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / sum)
}

function entropy(p: number[]): number {
  let h = 0
  for (const v of p) if (v > 0) h -= v * Math.log2(v)
  return h
}

export default function TemperatureRegimes() {
  const [T, setT] = useState(1.0)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const curve = useMemo(() => {
    const pts: Array<{ T: number; H: number }> = []
    const N = 200
    for (let i = 0; i <= N; i++) {
      const Ti = T_MIN * Math.pow(T_MAX / T_MIN, i / N)
      pts.push({ T: Ti, H: entropy(softmax(LOGITS, Ti)) })
    }
    return pts
  }, [])

  const probs = useMemo(() => softmax(LOGITS, T), [T])
  const H = entropy(probs)
  const Hmax = Math.log2(LOGITS.length)

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

      // Log-x scale for T, linear y for entropy.
      const toSx = (Tv: number) =>
        padL + (Math.log(Tv / T_MIN) / Math.log(T_MAX / T_MIN)) * plotW
      const toSy = (Hv: number) => padT + plotH - (Hv / (Hmax + 0.2)) * plotH

      // Grid
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.textAlign = 'center'
      ;[0.1, 0.3, 1, 3, 5].forEach((tv) => {
        const sx = toSx(tv)
        ctx.beginPath()
        ctx.moveTo(sx, padT)
        ctx.lineTo(sx, padT + plotH)
        ctx.stroke()
        ctx.fillText(String(tv), sx, padT + plotH + 14)
      })
      ctx.textAlign = 'right'
      ;[0, 1, Hmax].forEach((hv) => {
        const sy = toSy(hv)
        ctx.strokeStyle = '#1e1e1e'
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillStyle = '#555'
        ctx.fillText(hv.toFixed(2), padL - 6, sy + 3)
      })

      // Hmax dashed
      ctx.strokeStyle = '#444'
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(padL, toSy(Hmax))
      ctx.lineTo(padL + plotW, toSy(Hmax))
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = '#666'
      ctx.textAlign = 'left'
      ctx.fillText(`log₂(${LOGITS.length}) = ${Hmax.toFixed(2)}`, padL + plotW - 90, toSy(Hmax) - 5)

      // Curve
      ctx.strokeStyle = '#a78bfa'
      ctx.lineWidth = 2
      ctx.beginPath()
      curve.forEach((pt, i) => {
        const sx = toSx(pt.T)
        const sy = toSy(pt.H)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.stroke()

      // Cursor
      const cx = toSx(T)
      const cy = toSy(H)
      ctx.strokeStyle = '#fbbf24'
      ctx.setLineDash([2, 3])
      ctx.beginPath()
      ctx.moveTo(cx, padT)
      ctx.lineTo(cx, padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = '#fbbf24'
      ctx.beginPath()
      ctx.arc(cx, cy, 4, 0, Math.PI * 2)
      ctx.fill()

      // X axis label
      ctx.fillStyle = '#777'
      ctx.textAlign = 'center'
      ctx.fillText('temperature T (log)', padL + plotW / 2, padT + plotH + 22)
    }

    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [curve, T, H, Hmax])

  const regime = T < 0.3 ? 'peaked — near one-hot' : T > 3 ? 'diffuse — near uniform' : 'typical'

  return (
    <WidgetFrame
      widgetName="TemperatureRegimes"
      label="temperature — from one-hot to uniform"
      right={<span className="font-mono">fixed logits · T sweeps</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="T"
            value={T}
            min={T_MIN}
            max={T_MAX}
            step={0.01}
            onChange={setT}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="regime" value={regime} accent="text-term-amber" />
            <Readout label="entropy" value={H.toFixed(3)} />
            <Readout label="H / Hmax" value={(H / Hmax).toFixed(2)} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 grid grid-cols-[1fr_1.2fr] gap-4 p-5">
        <div className="flex flex-col min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
            probabilities
          </div>
          <div className="flex-1 flex flex-col gap-2 justify-center">
            {probs.map((p, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className="w-6 text-[11px] font-mono text-dark-text-secondary">
                  {CLASSES[i]}
                </span>
                <div className="flex-1 h-5 bg-dark-surface-elevated/40 rounded overflow-hidden">
                  <div
                    className={cn(
                      'h-full transition-all',
                      p === Math.max(...probs) ? 'bg-term-amber/70' : 'bg-term-amber/30'
                    )}
                    style={{ width: `${p * 100}%` }}
                  />
                </div>
                <span className="w-10 text-right font-mono text-[10px] tabular-nums text-dark-text-muted">
                  {(p * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        <div ref={boxRef} className="relative min-h-0">
          <canvas ref={canvasRef} className="w-full h-full block" />
        </div>
      </div>
    </WidgetFrame>
  )
}
