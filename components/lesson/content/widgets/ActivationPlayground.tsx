'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Five activations, each with its analytical derivative. Keep the math explicit
// so the visual and the formula stay in lockstep.
type ActName = 'sigmoid' | 'tanh' | 'relu' | 'leaky-relu' | 'gelu'

interface ActDef {
  name: ActName
  label: string
  f: (x: number) => number
  df: (x: number) => number
  formula: string
  dFormula: string
  range: [number, number]
  note: string
}

const acts: Record<ActName, ActDef> = {
  sigmoid: {
    name: 'sigmoid',
    label: 'sigmoid',
    f: (x) => 1 / (1 + Math.exp(-x)),
    df: (x) => {
      const s = 1 / (1 + Math.exp(-x))
      return s * (1 - s)
    },
    formula: 'σ(x) = 1 / (1 + e⁻ˣ)',
    dFormula: "σ'(x) = σ(x)·(1 − σ(x))",
    range: [0, 1],
    note: 'Smooth. Squashed. Saturates — derivative vanishes in the tails.',
  },
  tanh: {
    name: 'tanh',
    label: 'tanh',
    f: (x) => Math.tanh(x),
    df: (x) => 1 - Math.tanh(x) ** 2,
    formula: 'tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)',
    dFormula: "tanh'(x) = 1 − tanh²(x)",
    range: [-1, 1],
    note: 'Zero-centered sigmoid cousin. Same saturation problem.',
  },
  relu: {
    name: 'relu',
    label: 'ReLU',
    f: (x) => Math.max(0, x),
    df: (x) => (x > 0 ? 1 : 0),
    formula: 'ReLU(x) = max(0, x)',
    dFormula: "ReLU'(x) = { 1 if x>0 else 0 }",
    range: [0, 5],
    note: 'Linear on the right, zero on the left. Trivial to compute, no saturation on the positive side.',
  },
  'leaky-relu': {
    name: 'leaky-relu',
    label: 'Leaky ReLU',
    f: (x) => (x > 0 ? x : 0.1 * x),
    df: (x) => (x > 0 ? 1 : 0.1),
    formula: 'LReLU(x) = { x if x>0 else 0.1·x }',
    dFormula: "LReLU'(x) = { 1 if x>0 else 0.1 }",
    range: [-0.5, 5],
    note: 'Small slope on the left keeps gradients alive through dead zones.',
  },
  gelu: {
    name: 'gelu',
    label: 'GELU',
    f: (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))),
    df: (x) => {
      // Numerical derivative — good enough for a plot; an exact form exists but is noisy.
      const h = 1e-4
      const f = acts.gelu.f
      return (f(x + h) - f(x - h)) / (2 * h)
    },
    formula: 'GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))',
    dFormula: 'derivative — smooth transition between 0 and 1',
    range: [-0.2, 5],
    note: 'The transformer-era default. Smooth ReLU with a dip near zero.',
  },
}

const X_MIN = -5
const X_MAX = 5
const Y_MIN = -1.2
const Y_MAX = 1.8

export default function ActivationPlayground() {
  const [active, setActive] = useState<ActName>('sigmoid')
  const [x, setX] = useState(0.6)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const def = acts[active]

  const samples = useMemo(() => {
    const out: Array<{ x: number; y: number; dy: number }> = []
    const N = 400
    for (let i = 0; i <= N; i++) {
      const xi = X_MIN + (i / N) * (X_MAX - X_MIN)
      out.push({ x: xi, y: def.f(xi), dy: def.df(xi) })
    }
    return out
  }, [def])

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

      const pad = 36
      const plotW = w - pad * 2
      const plotH = h - pad * 2
      const toSx = (xv: number) => pad + ((xv - X_MIN) / (X_MAX - X_MIN)) * plotW
      const toSy = (yv: number) => pad + plotH - ((yv - Y_MIN) / (Y_MAX - Y_MIN)) * plotH

      // Axes
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(pad, toSy(0))
      ctx.lineTo(pad + plotW, toSy(0))
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(toSx(0), pad)
      ctx.lineTo(toSx(0), pad + plotH)
      ctx.stroke()

      // Grid labels
      ctx.fillStyle = '#555'
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.textAlign = 'center'
      ;[-4, -2, 2, 4].forEach((xv) => ctx.fillText(String(xv), toSx(xv), pad + plotH + 14))
      ctx.textAlign = 'right'
      ;[-1, 1].forEach((yv) => ctx.fillText(String(yv), pad - 6, toSy(yv) + 3))

      // f(x) — main curve
      ctx.strokeStyle = '#a78bfa'
      ctx.lineWidth = 2
      ctx.beginPath()
      samples.forEach((s, i) => {
        const sx = toSx(s.x)
        const sy = toSy(Math.max(Y_MIN, Math.min(Y_MAX, s.y)))
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.stroke()

      // f'(x) — derivative, dashed amber
      ctx.strokeStyle = '#fbbf24'
      ctx.lineWidth = 1.5
      ctx.setLineDash([4, 3])
      ctx.beginPath()
      samples.forEach((s, i) => {
        const sx = toSx(s.x)
        const sy = toSy(Math.max(Y_MIN, Math.min(Y_MAX, s.dy)))
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.stroke()
      ctx.setLineDash([])

      // Cursor
      const cursorX = toSx(x)
      const yv = def.f(x)
      const dy = def.df(x)
      ctx.strokeStyle = '#444'
      ctx.lineWidth = 1
      ctx.setLineDash([2, 3])
      ctx.beginPath()
      ctx.moveTo(cursorX, pad)
      ctx.lineTo(cursorX, pad + plotH)
      ctx.stroke()
      ctx.setLineDash([])

      // Dot on f(x)
      ctx.fillStyle = '#a78bfa'
      ctx.beginPath()
      ctx.arc(cursorX, toSy(yv), 5, 0, Math.PI * 2)
      ctx.fill()

      // Dot on f'(x)
      ctx.fillStyle = '#fbbf24'
      ctx.beginPath()
      ctx.arc(cursorX, toSy(dy), 4, 0, Math.PI * 2)
      ctx.fill()

      // Legend
      ctx.font = '11px "JetBrains Mono", monospace'
      ctx.textAlign = 'left'
      ctx.fillStyle = '#a78bfa'
      ctx.fillText('— f(x)', pad + 4, pad + 14)
      ctx.fillStyle = '#fbbf24'
      ctx.fillText("– – f'(x)", pad + 60, pad + 14)
    }

    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [samples, x, def])

  const curY = def.f(x)
  const curDy = def.df(x)

  return (
    <WidgetFrame
      widgetName="ActivationPlayground"
      label="activation playground"
      right={
        <>
          <span className="font-mono">{def.formula}</span>
          <span className="text-dark-text-disabled">·</span>
          <span className="font-mono">{def.dFormula}</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1 flex-wrap">
            {(Object.keys(acts) as ActName[]).map((key) => (
              <button
                key={key}
                onClick={() => setActive(key)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono uppercase tracking-wider transition-all',
                  active === key
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary hover:border-dark-border-hover'
                )}
              >
                {acts[key].label}
              </button>
            ))}
          </div>
          <Slider
            label="x"
            value={x}
            min={-5}
            max={5}
            step={0.01}
            onChange={setX}
            accent="accent-dark-accent"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="f(x)" value={curY.toFixed(4)} accent="text-term-purple" />
            <Readout label="f'(x)" value={curDy.toFixed(4)} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div ref={boxRef} className="absolute inset-0">
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>
      <div className="absolute bottom-2 left-4 right-4 text-[11px] font-mono text-dark-text-muted pointer-events-none">
        {def.note}
      </div>
    </WidgetFrame>
  )
}
