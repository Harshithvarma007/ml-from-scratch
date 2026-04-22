'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Two-input, one-neuron classifier in 2D. We shade the input plane by the
// neuron's output — that's the decision surface. Three preset datasets (AND,
// OR, XOR) make the "can't solve XOR" claim visible, not rhetorical.

type Preset = 'and' | 'or' | 'xor'

const presets: Record<Preset, Array<{ x: number; y: number; label: 0 | 1 }>> = {
  and: [
    { x: 0, y: 0, label: 0 },
    { x: 1, y: 0, label: 0 },
    { x: 0, y: 1, label: 0 },
    { x: 1, y: 1, label: 1 },
  ],
  or: [
    { x: 0, y: 0, label: 0 },
    { x: 1, y: 0, label: 1 },
    { x: 0, y: 1, label: 1 },
    { x: 1, y: 1, label: 1 },
  ],
  xor: [
    { x: 0, y: 0, label: 0 },
    { x: 1, y: 0, label: 1 },
    { x: 0, y: 1, label: 1 },
    { x: 1, y: 1, label: 0 },
  ],
}

// One step of best-fit parameters using the analytic perceptron rule — we just
// seed sensible initial weights per dataset so the reader starts somewhere
// plausible.
const seeds: Record<Preset, { w1: number; w2: number; b: number }> = {
  and: { w1: 3, w2: 3, b: -4.5 },
  or: { w1: 3, w2: 3, b: -1.5 },
  xor: { w1: 0, w2: 0, b: 0 },
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z))
}

function classify(x: number, y: number, w1: number, w2: number, b: number): number {
  return sigmoid(w1 * x + w2 * y + b)
}

export default function DecisionBoundary2D() {
  const [preset, setPreset] = useState<Preset>('and')
  const [w1, setW1] = useState(seeds.and.w1)
  const [w2, setW2] = useState(seeds.and.w2)
  const [b, setB] = useState(seeds.and.b)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const data = presets[preset]

  const accuracy = useMemo(() => {
    const correct = data.filter((p) => {
      const pred = classify(p.x, p.y, w1, w2, b) > 0.5 ? 1 : 0
      return pred === p.label
    }).length
    return correct / data.length
  }, [data, w1, w2, b])

  const loss = useMemo(() => {
    let L = 0
    for (const p of data) {
      const q = classify(p.x, p.y, w1, w2, b)
      L += -(p.label * Math.log(Math.max(q, 1e-9)) + (1 - p.label) * Math.log(Math.max(1 - q, 1e-9)))
    }
    return L / data.length
  }, [data, w1, w2, b])

  useEffect(() => {
    const c = canvasRef.current
    const box = boxRef.current
    if (!c || !box) return
    const dpr = window.devicePixelRatio || 1
    const draw = () => {
      const w = box.clientWidth
      const h = box.clientHeight
      c.width = w * dpr
      c.height = h * dpr
      c.style.width = `${w}px`
      c.style.height = `${h}px`
      const ctx = c.getContext('2d')!
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, w, h)

      const pad = 28
      const plotW = w - pad * 2
      const plotH = h - pad * 2
      const X_MIN = -0.3
      const X_MAX = 1.3
      const toSx = (xv: number) => pad + ((xv - X_MIN) / (X_MAX - X_MIN)) * plotW
      const toSy = (yv: number) => pad + plotH - ((yv - X_MIN) / (X_MAX - X_MIN)) * plotH

      // Paint the decision surface as a coarse heatmap — good enough for 2D.
      const STEP = 5
      const imgData = ctx.createImageData(plotW, plotH)
      for (let py = 0; py < plotH; py += STEP) {
        for (let px = 0; px < plotW; px += STEP) {
          const xv = X_MIN + (px / plotW) * (X_MAX - X_MIN)
          const yv = X_MIN + (1 - py / plotH) * (X_MAX - X_MIN)
          const q = classify(xv, yv, w1, w2, b)
          // Blend between rose (class 0) and cyan (class 1)
          const r = q < 0.5 ? 244 : 103
          const g = q < 0.5 ? 114 : 232
          const b8 = q < 0.5 ? 182 : 249
          const alpha = Math.abs(q - 0.5) * 2 * 80 + 20
          for (let dy = 0; dy < STEP && py + dy < plotH; dy++) {
            for (let dx = 0; dx < STEP && px + dx < plotW; dx++) {
              const idx = ((py + dy) * plotW + (px + dx)) * 4
              imgData.data[idx] = r
              imgData.data[idx + 1] = g
              imgData.data[idx + 2] = b8
              imgData.data[idx + 3] = alpha
            }
          }
        }
      }
      ctx.putImageData(imgData, pad, pad)

      // Decision line w1*x + w2*y + b = 0  →  y = -(w1*x + b)/w2
      ctx.strokeStyle = '#ffffff'
      ctx.globalAlpha = 0.55
      ctx.lineWidth = 1.5
      ctx.beginPath()
      if (Math.abs(w2) > 1e-4) {
        const yLeft = -(w1 * X_MIN + b) / w2
        const yRight = -(w1 * X_MAX + b) / w2
        ctx.moveTo(toSx(X_MIN), toSy(yLeft))
        ctx.lineTo(toSx(X_MAX), toSy(yRight))
      } else if (Math.abs(w1) > 1e-4) {
        const xLine = -b / w1
        ctx.moveTo(toSx(xLine), toSy(X_MIN))
        ctx.lineTo(toSx(xLine), toSy(X_MAX))
      }
      ctx.stroke()
      ctx.globalAlpha = 1

      // Grid
      ctx.strokeStyle = '#333'
      ctx.lineWidth = 1
      ;[0, 1].forEach((v) => {
        ctx.beginPath()
        ctx.moveTo(toSx(v), pad)
        ctx.lineTo(toSx(v), pad + plotH)
        ctx.stroke()
        ctx.beginPath()
        ctx.moveTo(pad, toSy(v))
        ctx.lineTo(pad + plotW, toSy(v))
        ctx.stroke()
      })

      // Data points
      data.forEach((p) => {
        const pred = classify(p.x, p.y, w1, w2, b) > 0.5 ? 1 : 0
        const correct = pred === p.label
        ctx.fillStyle = p.label === 1 ? '#67e8f9' : '#f472b6'
        ctx.strokeStyle = correct ? '#ffffff' : '#f87171'
        ctx.lineWidth = correct ? 2 : 3
        ctx.beginPath()
        ctx.arc(toSx(p.x), toSy(p.y), 10, 0, Math.PI * 2)
        ctx.fill()
        ctx.stroke()
        // Label inside the dot
        ctx.fillStyle = '#0a0a0a'
        ctx.font = 'bold 10px "JetBrains Mono", monospace'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(String(p.label), toSx(p.x), toSy(p.y))
      })

      // Axis labels
      ctx.fillStyle = '#666'
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillText('x₁', toSx(1), toSy(0) + 14)
      ctx.textAlign = 'right'
      ctx.fillText('x₂', toSx(0) - 8, toSy(1))
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [data, w1, w2, b])

  const setPresetAndReset = (p: Preset) => {
    setPreset(p)
    setW1(seeds[p].w1)
    setW2(seeds[p].w2)
    setB(seeds[p].b)
  }

  return (
    <WidgetFrame
      widgetName="DecisionBoundary2D"
      label="decision boundary — one neuron draws one line"
      right={
        <>
          <span className="font-mono">w₁x₁ + w₂x₂ + b = 0</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            <span className="text-[11px] font-mono text-dark-text-disabled uppercase tracking-wider mr-1">
              dataset
            </span>
            {(['and', 'or', 'xor'] as Preset[]).map((p) => (
              <button
                key={p}
                onClick={() => setPresetAndReset(p)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono uppercase transition-all',
                  preset === p
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
                )}
              >
                {p}
              </button>
            ))}
          </div>
          <Slider label="w₁" value={w1} min={-5} max={5} step={0.1} onChange={setW1} accent="accent-term-purple" />
          <Slider label="w₂" value={w2} min={-5} max={5} step={0.1} onChange={setW2} accent="accent-term-purple" />
          <Slider label="b" value={b} min={-6} max={2} step={0.1} onChange={setB} accent="accent-term-green" />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="loss" value={loss.toFixed(3)} accent="text-term-amber" />
            <Readout
              label="accuracy"
              value={`${(accuracy * 100).toFixed(0)}%`}
              accent={accuracy === 1 ? 'text-term-green' : 'text-term-rose'}
            />
          </div>
        </div>
      }
    >
      <div ref={boxRef} className="absolute inset-0">
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>
      {preset === 'xor' && accuracy < 1 && (
        <div className="absolute top-3 left-3 text-[11px] font-mono text-term-rose/90 bg-dark-bg/70 px-2 py-1 rounded border border-term-rose/40 pointer-events-none">
          XOR is not linearly separable · no line will ever score 100%
        </div>
      )}
    </WidgetFrame>
  )
}
