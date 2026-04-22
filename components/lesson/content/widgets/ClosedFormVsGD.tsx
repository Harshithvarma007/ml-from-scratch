'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Benchmark-style racing comparison. Varying N (number of examples) and D
// (number of features), we estimate both the runtime of the closed form
// (normal equations — O(N·D² + D³)) and the runtime of an iterative GD pass
// (O(iters · N · D)). Bars show who wins. Below a threshold, closed form is
// fastest and exact; above it, GD wins because the matrix inversion blows up.

const ITERS = 100 // typical iteration count for a well-tuned GD run

function closedFormCost(N: number, D: number): number {
  // Roughly the cost in "float multiplies" units: X^T X is N*D^2, invert is D^3.
  return N * D * D + D * D * D
}

function gdCost(N: number, D: number): number {
  return ITERS * N * D
}

// A made-up but honest runtime estimate (floats-per-second).
// We're not actually running it — the shape of the ratio is what matters.
const FLOPS = 1e9

function fmtSeconds(ops: number): string {
  const s = ops / FLOPS
  if (s < 1e-3) return (s * 1e6).toFixed(1) + ' µs'
  if (s < 1) return (s * 1e3).toFixed(1) + ' ms'
  if (s < 60) return s.toFixed(2) + ' s'
  if (s < 3600) return (s / 60).toFixed(1) + ' min'
  return (s / 3600).toFixed(1) + ' hr'
}

export default function ClosedFormVsGD() {
  const [N, setN] = useState(1_000)
  const [D, setD] = useState(10)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const costs = useMemo(() => {
    const cf = closedFormCost(N, D)
    const gd = gdCost(N, D)
    return { cf, gd, ratio: cf / gd }
  }, [N, D])

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

      // Sweep D from 1 to 2000, plot both costs at fixed N
      const padL = 56
      const padR = 12
      const padT = 18
      const padB = 32
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const D_MIN = 1
      const D_MAX = 2000

      // log-log
      const toSx = (Dv: number) =>
        padL + (Math.log(Dv / D_MIN) / Math.log(D_MAX / D_MIN)) * plotW
      const toSy = (Cv: number) => {
        const L = Math.log10(Math.max(Cv, 1))
        return padT + plotH - (L / 14) * plotH
      }

      // Axes / grid
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.textAlign = 'center'
      ;[1, 10, 100, 1000].forEach((dv) => {
        const sx = toSx(dv)
        ctx.beginPath()
        ctx.moveTo(sx, padT)
        ctx.lineTo(sx, padT + plotH)
        ctx.stroke()
        ctx.fillText(String(dv), sx, padT + plotH + 14)
      })
      ctx.textAlign = 'right'
      ;[1, 6, 9, 12].forEach((pw) => {
        const sy = toSy(Math.pow(10, pw))
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(`10${toSup(pw)}`, padL - 6, sy + 3)
      })

      // Closed form curve
      const drawCurve = (fn: (D: number) => number, color: string) => {
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.beginPath()
        for (let step = 0; step <= 200; step++) {
          const Dv = D_MIN * Math.pow(D_MAX / D_MIN, step / 200)
          const Cv = fn(Dv)
          const sx = toSx(Dv)
          const sy = toSy(Cv)
          if (step === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        ctx.stroke()
      }

      drawCurve((d) => closedFormCost(N, d), '#a78bfa')
      drawCurve((d) => gdCost(N, d), '#fbbf24')

      // Cursor at current D
      const sx = toSx(D)
      ctx.strokeStyle = '#555'
      ctx.setLineDash([3, 4])
      ctx.beginPath()
      ctx.moveTo(sx, padT)
      ctx.lineTo(sx, padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])

      // Dots
      ctx.fillStyle = '#a78bfa'
      ctx.beginPath()
      ctx.arc(sx, toSy(costs.cf), 5, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = '#fbbf24'
      ctx.beginPath()
      ctx.arc(sx, toSy(costs.gd), 5, 0, Math.PI * 2)
      ctx.fill()

      // Axis labels
      ctx.fillStyle = '#777'
      ctx.textAlign = 'center'
      ctx.fillText('features D  (log)', padL + plotW / 2, padT + plotH + 26)

      ctx.textAlign = 'left'
      ctx.fillStyle = '#a78bfa'
      ctx.fillRect(padL + 6, padT + 4, 10, 2)
      ctx.fillText('closed form  O(N·D² + D³)', padL + 20, padT + 8)
      ctx.fillStyle = '#fbbf24'
      ctx.fillRect(padL + 6, padT + 20, 10, 2)
      ctx.fillText(`gradient descent  O(${ITERS}·N·D)`, padL + 20, padT + 24)
    }

    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [N, D, costs])

  const winner = costs.cf < costs.gd ? 'closed form' : 'gradient descent'

  return (
    <WidgetFrame
      widgetName="ClosedFormVsGD"
      label="the regimes — when each algorithm wins"
      right={<span>assumes {ITERS}-iter GD · 1 GFLOP/s budget</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="N"
            value={N}
            min={10}
            max={1_000_000}
            step={100}
            onChange={(v) => setN(Math.round(v))}
            format={(v) => fmt(Math.round(v))}
            accent="accent-dark-accent"
          />
          <Slider
            label="D"
            value={D}
            min={1}
            max={2000}
            step={1}
            onChange={(v) => setD(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-purple"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="winner"
              value={winner}
              accent={winner === 'closed form' ? 'text-term-purple' : 'text-term-amber'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 grid grid-cols-[1.1fr_1fr] gap-4 p-4">
        <div ref={boxRef} className="relative min-h-0">
          <canvas ref={canvasRef} className="w-full h-full block" />
        </div>

        <div className="flex flex-col justify-center gap-4 px-2">
          <Bar
            label="closed form"
            value={costs.cf}
            maxValue={Math.max(costs.cf, costs.gd)}
            color="#a78bfa"
            subtitle={fmtSeconds(costs.cf) + ` · ${fmt(costs.cf)} ops`}
          />
          <Bar
            label="gradient descent"
            value={costs.gd}
            maxValue={Math.max(costs.cf, costs.gd)}
            color="#fbbf24"
            subtitle={fmtSeconds(costs.gd) + ` · ${fmt(costs.gd)} ops`}
          />
          <div className="text-[11px] font-mono text-dark-text-muted mt-2 leading-relaxed">
            {costs.cf < costs.gd
              ? 'Small D · closed form is both fastest and exact. This is the regime your intro stats course lives in.'
              : 'Large D · matrix inversion dominates. Iterative methods win by orders of magnitude — the entire deep learning industry is built on this fact.'}
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function Bar({
  label,
  value,
  maxValue,
  color,
  subtitle,
}: {
  label: string
  value: number
  maxValue: number
  color: string
  subtitle: string
}) {
  const pct = Math.min(100, (value / maxValue) * 100)
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-[11px] font-mono uppercase tracking-wider text-dark-text-primary">
          {label}
        </span>
        <span className={cn('text-[10px] font-mono')} style={{ color }}>
          {pct.toFixed(1)}%
        </span>
      </div>
      <div className="h-6 bg-dark-surface-elevated/40 rounded overflow-hidden">
        <div
          className="h-full transition-all"
          style={{ width: `${pct}%`, backgroundColor: color + 'aa' }}
        />
      </div>
      <div className="text-[10.5px] font-mono text-dark-text-disabled mt-1">{subtitle}</div>
    </div>
  )
}

function fmt(n: number): string {
  if (n < 1e4) return String(n)
  if (n < 1e6) return (n / 1e3).toFixed(1) + 'K'
  if (n < 1e9) return (n / 1e6).toFixed(1) + 'M'
  if (n < 1e12) return (n / 1e9).toFixed(1) + 'B'
  return (n / 1e12).toFixed(1) + 'T'
}

function toSup(n: number): string {
  const map = '⁰¹²³⁴⁵⁶⁷⁸⁹'
  return String(n)
    .split('')
    .map((d) => map[Number(d)] ?? d)
    .join('')
}
