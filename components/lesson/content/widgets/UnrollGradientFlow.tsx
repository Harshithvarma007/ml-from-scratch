'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// For an unrolled RNN of length N, plot |∂L/∂h_t| as a function of t (with
// t = N at the right, representing the loss side, and t = 0 at the left,
// representing the distant past). The slope of the curve is controlled by the
// spectral radius σ = ‖W_h‖. σ < 1 → vanishing, σ > 1 → exploding, σ = 1 →
// flat. The widget lets the user drag σ and N; a dotted overlay shows the
// ideal theoretical decay σ^(N - t) for comparison.

const MAX_N = 120

function gradientTrace(N: number, sigma: number): number[] {
  // We start at t = N with |grad| = 1 and walk backwards. At each step we
  // multiply by |tanh'(z)| · σ, where tanh'(z) is modeled as an average 0.8
  // (a realistic value for lightly active tanh units). Only σ is user-driven;
  // everything else is implicit in the constant.
  const avgDeriv = 0.8
  const out: number[] = new Array(N + 1).fill(0)
  out[N] = 1
  for (let t = N - 1; t >= 0; t--) {
    out[t] = out[t + 1] * avgDeriv * sigma
  }
  return out
}

export default function UnrollGradientFlow() {
  const [N, setN] = useState(40)
  const [sigma, setSigma] = useState(0.9)
  const trace = useMemo(() => gradientTrace(N, sigma), [N, sigma])
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

      const padL = 56
      const padR = 24
      const padT = 22
      const padB = 32
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const toSx = (t: number) => padL + (t / MAX_N) * plotW
      const yMinLog = -20
      const yMaxLog = 6
      const toSy = (v: number) => {
        const logV = Math.log10(Math.max(Math.abs(v), 1e-22))
        return padT + plotH - ((logV - yMinLog) / (yMaxLog - yMinLog)) * plotH
      }

      // Grid lines
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      for (let p = -18; p <= 6; p += 3) {
        const sy = toSy(Math.pow(10, p))
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(p === 0 ? '1' : `10${toSup(p)}`, padL - 6, sy + 3)
      }
      ctx.textAlign = 'center'
      ;[0, 30, 60, 90, 120].forEach((t) => ctx.fillText(String(t), toSx(t), padT + plotH + 14))
      ctx.fillStyle = '#777'
      ctx.fillText('time step t  (loss at right, past at left)', padL + plotW / 2, padT + plotH + 26)

      // Float32 underflow line (ish)
      const underflowSy = toSy(1e-7)
      ctx.strokeStyle = 'rgba(248, 113, 113, 0.25)'
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(padL, underflowSy)
      ctx.lineTo(padL + plotW, underflowSy)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.textAlign = 'left'
      ctx.fillStyle = 'rgba(248, 113, 113, 0.85)'
      ctx.fillText('≈ float32 precision floor', padL + 6, underflowSy - 4)

      // The trace
      const color = sigma < 1 ? '#f87171' : sigma > 1 ? '#fbbf24' : '#4ade80'
      ctx.strokeStyle = color
      ctx.lineWidth = 2.2
      ctx.beginPath()
      for (let t = 0; t <= N; t++) {
        const sx = toSx(t)
        const sy = toSy(trace[t])
        if (t === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()

      // Endpoints
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.arc(toSx(N), toSy(trace[N]), 4, 0, Math.PI * 2)
      ctx.fill()
      ctx.beginPath()
      ctx.arc(toSx(0), toSy(trace[0]), 4, 0, Math.PI * 2)
      ctx.fill()

      // Loss marker
      ctx.fillStyle = '#e5e7eb'
      ctx.textAlign = 'right'
      ctx.fillText('loss', toSx(N) - 6, toSy(trace[N]) - 8)
      ctx.textAlign = 'left'
      ctx.fillText(`|∂L/∂h_0| ≈ ${trace[0].toExponential(1)}`, toSx(0) + 6, toSy(trace[0]) - 8)

      // Annotation
      ctx.fillStyle = color
      ctx.font = 'bold 11px "JetBrains Mono", monospace'
      ctx.textAlign = 'left'
      ctx.fillText(
        sigma < 1
          ? 'VANISHING — gradient dies exponentially'
          : sigma > 1
            ? 'EXPLODING — gradient blows up exponentially'
            : 'MARGINAL — the unicorn case',
        padL + 8,
        padT + 14,
      )
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [trace, N, sigma])

  return (
    <WidgetFrame
      widgetName="UnrollGradientFlow"
      label="|∂L/∂h_t| — gradient magnitude backpropagating through an unrolled RNN"
      right={<span className="font-mono">|grad_t| ≈ (σ · &lt;tanh&apos;&gt;)^(N − t)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="σ_max(W_h)"
            value={sigma}
            min={0.3}
            max={1.4}
            step={0.01}
            onChange={setSigma}
            format={(v) => v.toFixed(2)}
            accent="accent-term-amber"
          />
          <Slider
            label="seq len N"
            value={N}
            min={5}
            max={MAX_N}
            step={1}
            onChange={(v) => setN(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-dark-accent"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="|∂L/∂h_0|"
              value={trace[0].toExponential(1)}
              accent={trace[0] < 1e-6 ? 'text-term-rose' : trace[0] > 1e3 ? 'text-term-amber' : 'text-term-green'}
            />
            <Readout
              label="regime"
              value={sigma < 0.98 ? 'vanishing' : sigma > 1.02 ? 'exploding' : 'marginal'}
              accent={sigma < 0.98 ? 'text-term-rose' : sigma > 1.02 ? 'text-term-amber' : 'text-term-green'}
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
