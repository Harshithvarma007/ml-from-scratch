'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Show why sigmoid dies in deep nets. Same signal, N layers, product of
// derivatives. Sigmoid derivative maxes out at 0.25 — stack 20 of those and
// your gradient shrinks by 10^-12. ReLU derivative is 1 where it's positive,
// so the product stays put. The chart makes it visceral.

type ActName = 'sigmoid' | 'tanh' | 'relu'

const colors: Record<ActName, string> = {
  sigmoid: '#a78bfa',
  tanh: '#60a5fa',
  relu: '#fbbf24',
}

const MAX_DEPTH = 30

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x))
}

// Expected |derivative| after feeding a signal through layer `k` of activation `act`,
// assuming the pre-activation is roughly Gaussian around zero (standard deviation ~1).
// We just estimate by sampling — good enough to show the regime.
function sampleMeanAbsDerivative(act: ActName, signalStd: number): number {
  const N = 400
  let sum = 0
  for (let i = 0; i < N; i++) {
    // Sample z ~ N(0, signalStd²) via Box–Muller.
    const u1 = (i + 0.5) / N
    const u2 = ((i * 17) % N) / N + 1e-6
    const r = Math.sqrt(-2 * Math.log(u1))
    const z = r * Math.cos(2 * Math.PI * u2) * signalStd
    let d = 0
    if (act === 'sigmoid') {
      const s = sigmoid(z)
      d = s * (1 - s)
    } else if (act === 'tanh') {
      d = 1 - Math.tanh(z) ** 2
    } else {
      d = z > 0 ? 1 : 0
    }
    sum += d
  }
  return sum / N
}

function computeChain(act: ActName, depth: number): number[] {
  // Track the product of expected |f'(z)| per layer. Start at 1.
  const gains: number[] = [1]
  const g = sampleMeanAbsDerivative(act, 1)
  for (let i = 1; i <= depth; i++) {
    gains.push(gains[i - 1] * g)
  }
  return gains
}

export default function SaturationChain() {
  const [depth, setDepth] = useState(20)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const series = useMemo(
    () => ({
      sigmoid: computeChain('sigmoid', MAX_DEPTH),
      tanh: computeChain('tanh', MAX_DEPTH),
      relu: computeChain('relu', MAX_DEPTH),
    }),
    [],
  )

  const final = {
    sigmoid: series.sigmoid[depth],
    tanh: series.tanh[depth],
    relu: series.relu[depth],
  }

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
      const padR = 16
      const padT = 24
      const padB = 32
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      // Log-scale y — that's the only way to see both regimes at once.
      const yMaxLog = 0 // log10(1) = 0
      const yMinLog = -14 // show all the way down

      const toSx = (k: number) => padL + (k / MAX_DEPTH) * plotW
      const toSy = (v: number) => {
        const L = Math.max(yMinLog, Math.log10(Math.max(v, 1e-20)))
        return padT + plotH - ((L - yMinLog) / (yMaxLog - yMinLog)) * plotH
      }

      // Grid + y labels (powers of 10)
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      ;[0, -3, -6, -9, -12].forEach((p) => {
        const y = toSy(Math.pow(10, p))
        ctx.strokeStyle = '#1e1e1e'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(padL, y)
        ctx.lineTo(padL + plotW, y)
        ctx.stroke()
        ctx.fillText(p === 0 ? '1' : `10${toSuper(p)}`, padL - 8, y + 3)
      })

      // X labels
      ctx.textAlign = 'center'
      ;[0, 10, 20, 30].forEach((k) => {
        ctx.fillText(String(k), toSx(k), padT + plotH + 14)
      })
      ctx.fillStyle = '#888'
      ctx.fillText('layer depth k', padL + plotW / 2, padT + plotH + 26)

      // Plot each curve
      const draw_series = (data: number[], color: string) => {
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.beginPath()
        data.forEach((v, k) => {
          if (k > depth) return
          const sx = toSx(k)
          const sy = toSy(v)
          if (k === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        ctx.stroke()
        // Faded continuation beyond current depth
        ctx.globalAlpha = 0.25
        ctx.setLineDash([3, 3])
        ctx.beginPath()
        data.forEach((v, k) => {
          if (k < depth) return
          const sx = toSx(k)
          const sy = toSy(v)
          if (k === depth) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        ctx.stroke()
        ctx.setLineDash([])
        ctx.globalAlpha = 1
      }

      draw_series(series.sigmoid, colors.sigmoid)
      draw_series(series.tanh, colors.tanh)
      draw_series(series.relu, colors.relu)

      // Cursor at current depth
      ctx.strokeStyle = '#3a3a3a'
      ctx.setLineDash([2, 4])
      ctx.beginPath()
      ctx.moveTo(toSx(depth), padT)
      ctx.lineTo(toSx(depth), padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])

      // Legend
      let lx = padL + 12
      const ly = padT + 16
      ;(['sigmoid', 'tanh', 'relu'] as ActName[]).forEach((a) => {
        ctx.fillStyle = colors[a]
        ctx.fillRect(lx, ly - 6, 10, 2)
        ctx.fillStyle = '#ccc'
        ctx.textAlign = 'left'
        ctx.fillText(a, lx + 14, ly)
        lx += 14 + ctx.measureText(a).width + 20
      })
    }

    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [series, depth])

  return (
    <WidgetFrame
      widgetName="SaturationChain"
      label="saturation — expected gradient magnitude through k layers"
      right={<span>log₁₀ scale · signal ~ N(0, 1)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="depth k"
            value={depth}
            min={1}
            max={MAX_DEPTH}
            step={1}
            onChange={setDepth}
            format={(n) => String(n).padStart(2, ' ')}
            accent="accent-dark-accent"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="sigmoid" value={fmt(final.sigmoid)} accent="text-term-purple" />
            <Readout label="tanh" value={fmt(final.tanh)} accent="text-dark-sql" />
            <Readout label="relu" value={fmt(final.relu)} accent="text-term-amber" />
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

function fmt(v: number): string {
  if (v < 1e-3) return v.toExponential(1)
  return v.toFixed(4)
}

function toSuper(n: number): string {
  const map: Record<string, string> = {
    '-': '⁻',
    '0': '⁰',
    '1': '¹',
    '2': '²',
    '3': '³',
    '4': '⁴',
    '5': '⁵',
    '6': '⁶',
    '7': '⁷',
    '8': '⁸',
    '9': '⁹',
  }
  return String(n)
    .split('')
    .map((c) => map[c] ?? c)
    .join('')
}
