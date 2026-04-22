'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// The canonical vanishing-gradient picture: |∂L/∂h_0| plotted as a function
// of sequence length N for three spectral radii. On a log y-axis the curves
// are straight lines, slopes equal to log(ρ · <tanh'>). The user can toggle
// between y-axes (linear shows the dead-zone visually; log shows the actual
// exponent). A vertical cursor shows the gradient magnitude at the current N.

type Activation = 'tanh' | 'sigmoid' | 'relu'

const MAX_N = 100

function avgDeriv(act: Activation): number {
  // Expected |f'(z)| under z ~ N(0, 1) — computed analytically / numerically
  if (act === 'tanh') return 0.79
  if (act === 'sigmoid') return 0.21
  return 0.5 // relu: half the time the derivative is 1, half 0
}

function trace(rho: number, act: Activation): number[] {
  const out: number[] = new Array(MAX_N + 1).fill(0)
  out[0] = 1
  const factor = rho * avgDeriv(act)
  for (let n = 1; n <= MAX_N; n++) out[n] = out[n - 1] * factor
  return out
}

const CURVES: { label: string; rho: number; color: string }[] = [
  { label: 'ρ = 0.6 · tight vanish', rho: 0.6, color: '#f87171' },
  { label: 'ρ = 0.95 · soft vanish', rho: 0.95, color: '#fbbf24' },
  { label: 'ρ = 1.05 · soft explode', rho: 1.05, color: '#a78bfa' },
  { label: 'ρ = 1.20 · fast explode', rho: 1.20, color: '#4ade80' },
]

export default function GradientOverTime() {
  const [N, setN] = useState(40)
  const [act, setAct] = useState<Activation>('tanh')
  const [logScale, setLogScale] = useState(true)

  const series = useMemo(
    () =>
      CURVES.map(({ label, rho, color }) => ({
        label,
        rho,
        color,
        data: trace(rho, act),
      })),
    [act],
  )
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

      const padL = 62
      const padR = 24
      const padT = 24
      const padB = 36
      const plotW = w - padL - padR
      const plotH = h - padT - padB
      const toSx = (n: number) => padL + (n / MAX_N) * plotW

      const yMinLog = -20
      const yMaxLog = 6
      const toSy = (v: number) => {
        if (logScale) {
          const logV = Math.log10(Math.max(Math.abs(v), 1e-22))
          return padT + plotH - ((logV - yMinLog) / (yMaxLog - yMinLog)) * plotH
        }
        const clamped = Math.max(-2, Math.min(5, v))
        return padT + plotH - ((clamped + 2) / 7) * plotH
      }

      // Gridlines & axis labels
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      if (logScale) {
        for (let p = -18; p <= 6; p += 3) {
          const sy = toSy(Math.pow(10, p))
          ctx.beginPath()
          ctx.moveTo(padL, sy)
          ctx.lineTo(padL + plotW, sy)
          ctx.stroke()
          ctx.fillText(p === 0 ? '1' : `10${toSup(p)}`, padL - 6, sy + 3)
        }
      } else {
        ;[-2, -1, 0, 1, 2, 3, 4, 5].forEach((v) => {
          const sy = toSy(v)
          ctx.beginPath()
          ctx.moveTo(padL, sy)
          ctx.lineTo(padL + plotW, sy)
          ctx.stroke()
          ctx.fillText(String(v), padL - 6, sy + 3)
        })
      }
      ctx.textAlign = 'center'
      ;[0, 20, 40, 60, 80, 100].forEach((n) => ctx.fillText(String(n), toSx(n), padT + plotH + 14))
      ctx.fillStyle = '#777'
      ctx.fillText('sequence length N', padL + plotW / 2, padT + plotH + 28)

      // Float floor
      if (logScale) {
        const sy = toSy(1e-7)
        ctx.strokeStyle = 'rgba(248,113,113,0.25)'
        ctx.setLineDash([4, 4])
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.setLineDash([])
      }

      // Curves
      series.forEach(({ color, data }) => {
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.beginPath()
        for (let n = 0; n <= MAX_N; n++) {
          const sx = toSx(n)
          const sy = toSy(data[n])
          if (n === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        ctx.stroke()
      })

      // Current-N cursor
      ctx.strokeStyle = 'rgba(255,255,255,0.25)'
      ctx.setLineDash([3, 4])
      ctx.beginPath()
      ctx.moveTo(toSx(N), padT)
      ctx.lineTo(toSx(N), padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])

      series.forEach(({ color, data }) => {
        ctx.fillStyle = color
        ctx.beginPath()
        ctx.arc(toSx(N), toSy(data[N]), 4, 0, Math.PI * 2)
        ctx.fill()
      })

      // Legend
      let lx = padL + 10
      const ly = padT + 16
      series.forEach(({ label, color }) => {
        ctx.fillStyle = color
        ctx.fillRect(lx, ly - 6, 10, 2)
        ctx.fillStyle = '#ccc'
        ctx.textAlign = 'left'
        ctx.fillText(label, lx + 14, ly)
        lx += 14 + ctx.measureText(label).width + 18
      })
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [series, N, logScale])

  return (
    <WidgetFrame
      widgetName="GradientOverTime"
      label="gradient magnitude vs. sequence length — log collapse in action"
      right={<span className="font-mono">|∂L/∂h_0| ≈ (ρ · 〈f&apos;〉)^N</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="N"
            value={N}
            min={1}
            max={MAX_N}
            step={1}
            onChange={(v) => setN(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-1">
            {(['tanh', 'sigmoid', 'relu'] as Activation[]).map((a) => (
              <button
                key={a}
                onClick={() => setAct(a)}
                className={cn(
                  'px-2 py-1 rounded text-[10.5px] font-mono uppercase transition-all',
                  act === a ? 'bg-dark-accent text-white' : 'border border-dark-border text-dark-text-secondary',
                )}
              >
                {a}
              </button>
            ))}
          </div>
          <button
            onClick={() => setLogScale((v) => !v)}
            className={cn(
              'px-2 py-1 rounded text-[10.5px] font-mono uppercase transition-all',
              logScale ? 'bg-dark-accent text-white' : 'border border-dark-border text-dark-text-secondary',
            )}
          >
            log y
          </button>
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="ρ=0.95"
              value={series[1].data[N].toExponential(1)}
              accent="text-term-amber"
            />
            <Readout
              label="ρ=1.05"
              value={series[2].data[N].toExponential(1)}
              accent="text-term-purple"
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

function toSup(n: number): string {
  const map = '⁰¹²³⁴⁵⁶⁷⁸⁹'
  return String(n)
    .split('')
    .map((c) => (c === '-' ? '⁻' : map[Number(c)] ?? c))
    .join('')
}
