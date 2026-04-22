'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Two side-by-side plots. Left: the input space with XOR data (not linearly
// separable). Right: the hidden-layer activation space — where the same
// points land after one ReLU layer. Drag the hidden-layer weights; watch the
// four corners shuffle until they become linearly separable. Hidden layers =
// learned coordinate systems.

// The weights below are tuned so the default state already warps XOR into a
// separable configuration — this lets the reader start in the "solved" state
// and break it by dragging.

type Point = { x: number; y: number; label: 0 | 1 }

const DATA: Point[] = [
  { x: 0, y: 0, label: 0 },
  { x: 1, y: 0, label: 1 },
  { x: 0, y: 1, label: 1 },
  { x: 1, y: 1, label: 0 },
]

function relu(z: number) {
  return Math.max(0, z)
}

export default function FeatureWarp() {
  // Hidden layer has 2 neurons. Parameters: W (2×2) + b (2).
  // Defaults: h1 detects "both on", h2 detects "both off".
  const [w11, setW11] = useState(1)
  const [w12, setW12] = useState(1)
  const [w21, setW21] = useState(-1)
  const [w22, setW22] = useState(-1)
  const [b1, setB1] = useState(-1.5)
  const [b2, setB2] = useState(0.5)

  const warped = useMemo(
    () =>
      DATA.map((p) => ({
        label: p.label,
        h1: relu(w11 * p.x + w12 * p.y + b1),
        h2: relu(w21 * p.x + w22 * p.y + b2),
        // Original
        x: p.x,
        y: p.y,
      })),
    [w11, w12, w21, w22, b1, b2],
  )

  const lCanvas = useRef<HTMLCanvasElement | null>(null)
  const rCanvas = useRef<HTMLCanvasElement | null>(null)
  const lBox = useRef<HTMLDivElement | null>(null)
  const rBox = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const drawPane = (
      canvas: HTMLCanvasElement | null,
      box: HTMLDivElement | null,
      points: Array<{ x: number; y: number; label: 0 | 1 }>,
      range: [number, number],
      xLabel: string,
      yLabel: string,
    ) => {
      if (!canvas || !box) return
      const dpr = window.devicePixelRatio || 1
      const w = box.clientWidth
      const h = box.clientHeight
      canvas.width = w * dpr
      canvas.height = h * dpr
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      const ctx = canvas.getContext('2d')!
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, w, h)

      const pad = 28
      const plotW = w - pad * 2
      const plotH = h - pad * 2
      const [lo, hi] = range
      const toSx = (v: number) => pad + ((v - lo) / (hi - lo)) * plotW
      const toSy = (v: number) => pad + plotH - ((v - lo) / (hi - lo)) * plotH

      // Axes
      ctx.strokeStyle = '#2a2a2a'
      ctx.beginPath()
      ctx.moveTo(pad, toSy(0))
      ctx.lineTo(pad + plotW, toSy(0))
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(toSx(0), pad)
      ctx.lineTo(toSx(0), pad + plotH)
      ctx.stroke()

      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#666'
      ctx.textAlign = 'center'
      ctx.fillText(xLabel, pad + plotW / 2, h - 4)
      ctx.save()
      ctx.translate(12, pad + plotH / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.fillText(yLabel, 0, 0)
      ctx.restore()

      // If we're on the right pane (hidden space), try to draw the best
      // separating line between the two classes. Not essential — but makes
      // "linearly separable" visible.
      if (xLabel === 'h₁' && points.length === 4) {
        const zeros = points.filter((p) => p.label === 0)
        const ones = points.filter((p) => p.label === 1)
        const mean0 = {
          x: zeros.reduce((s, p) => s + p.x, 0) / zeros.length,
          y: zeros.reduce((s, p) => s + p.y, 0) / zeros.length,
        }
        const mean1 = {
          x: ones.reduce((s, p) => s + p.x, 0) / ones.length,
          y: ones.reduce((s, p) => s + p.y, 0) / ones.length,
        }
        const dx = mean1.x - mean0.x
        const dy = mean1.y - mean0.y
        const mx = (mean0.x + mean1.x) / 2
        const my = (mean0.y + mean1.y) / 2
        // Line normal to (dx, dy) through (mx, my): dx·x + dy·y = dx·mx + dy·my
        const C = dx * mx + dy * my
        if (Math.abs(dy) > 1e-4) {
          ctx.strokeStyle = '#4ade80'
          ctx.setLineDash([4, 4])
          ctx.lineWidth = 1.5
          ctx.beginPath()
          const yL = (C - dx * lo) / dy
          const yR = (C - dx * hi) / dy
          ctx.moveTo(toSx(lo), toSy(yL))
          ctx.lineTo(toSx(hi), toSy(yR))
          ctx.stroke()
          ctx.setLineDash([])
        }
      }

      // Points
      points.forEach((p) => {
        ctx.fillStyle = p.label === 1 ? '#67e8f9' : '#f472b6'
        ctx.strokeStyle = '#0a0a0a'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(toSx(p.x), toSy(p.y), 8, 0, Math.PI * 2)
        ctx.fill()
        ctx.stroke()
        ctx.fillStyle = '#0a0a0a'
        ctx.font = 'bold 9px "JetBrains Mono", monospace'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(String(p.label), toSx(p.x), toSy(p.y))
      })
    }

    drawPane(lCanvas.current, lBox.current, DATA, [-0.3, 1.3], 'x₁', 'x₂')
    drawPane(
      rCanvas.current,
      rBox.current,
      warped.map((p) => ({ x: p.h1, y: p.h2, label: p.label })),
      [-0.3, 2.3],
      'h₁',
      'h₂',
    )

    const ros = [
      lBox.current && new ResizeObserver(() => drawPane(lCanvas.current, lBox.current, DATA, [-0.3, 1.3], 'x₁', 'x₂')),
      rBox.current &&
        new ResizeObserver(() =>
          drawPane(
            rCanvas.current,
            rBox.current,
            warped.map((p) => ({ x: p.h1, y: p.h2, label: p.label })),
            [-0.3, 2.3],
            'h₁',
            'h₂',
          ),
        ),
    ]
    if (lBox.current && ros[0]) ros[0].observe(lBox.current)
    if (rBox.current && ros[1]) ros[1].observe(rBox.current)
    return () => {
      ros.forEach((ro) => ro?.disconnect())
    }
  }, [warped])

  // Check if XOR is linearly separable in hidden space (simple: can a line put
  // the two class means on opposite sides of all points of the other class).
  const separable = useMemo(() => {
    const zeros = warped.filter((p) => p.label === 0)
    const ones = warped.filter((p) => p.label === 1)
    if (zeros.length < 2 || ones.length < 2) return false
    const mean0 = {
      x: zeros.reduce((s, p) => s + p.h1, 0) / zeros.length,
      y: zeros.reduce((s, p) => s + p.h2, 0) / zeros.length,
    }
    const mean1 = {
      x: ones.reduce((s, p) => s + p.h1, 0) / ones.length,
      y: ones.reduce((s, p) => s + p.h2, 0) / ones.length,
    }
    const dx = mean1.x - mean0.x
    const dy = mean1.y - mean0.y
    const mx = (mean0.x + mean1.x) / 2
    const my = (mean0.y + mean1.y) / 2
    const side = (p: { h1: number; h2: number }) => dx * (p.h1 - mx) + dy * (p.h2 - my)
    return (
      zeros.every((p) => side(p) < 0) && ones.every((p) => side(p) > 0) ||
      zeros.every((p) => side(p) > 0) && ones.every((p) => side(p) < 0)
    )
  }, [warped])

  return (
    <WidgetFrame
      widgetName="FeatureWarp"
      label="hidden layers warp the input space"
      right={<span className="font-mono">h = ReLU(W · x + b)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider label="W₁₁" value={w11} min={-2} max={2} step={0.05} onChange={setW11} accent="accent-term-purple" />
          <Slider label="W₁₂" value={w12} min={-2} max={2} step={0.05} onChange={setW12} accent="accent-term-purple" />
          <Slider label="W₂₁" value={w21} min={-2} max={2} step={0.05} onChange={setW21} accent="accent-dark-accent" />
          <Slider label="W₂₂" value={w22} min={-2} max={2} step={0.05} onChange={setW22} accent="accent-dark-accent" />
          <Slider label="b₁" value={b1} min={-2} max={2} step={0.05} onChange={setB1} accent="accent-term-green" />
          <Slider label="b₂" value={b2} min={-2} max={2} step={0.05} onChange={setB2} accent="accent-term-green" />
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="hidden-space separable"
              value={separable ? 'yes' : 'no'}
              accent={separable ? 'text-term-green' : 'text-term-rose'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 grid grid-cols-2 gap-2 p-2">
        <div className="border border-dark-border rounded-md overflow-hidden bg-dark-bg flex flex-col">
          <div className="px-3 py-1.5 border-b border-dark-border bg-dark-surface-elevated/40 flex items-center justify-between">
            <span className="text-[11px] font-mono uppercase tracking-wider">input space</span>
            <span className="text-[10px] font-mono text-term-rose">XOR · not linearly separable</span>
          </div>
          <div ref={lBox} className="flex-1 min-h-0 relative">
            <canvas ref={lCanvas} className="w-full h-full block" />
          </div>
        </div>
        <div className="border border-dark-border rounded-md overflow-hidden bg-dark-bg flex flex-col">
          <div className="px-3 py-1.5 border-b border-dark-border bg-dark-surface-elevated/40 flex items-center justify-between">
            <span className="text-[11px] font-mono uppercase tracking-wider">hidden space (h₁, h₂)</span>
            <span
              className={`text-[10px] font-mono ${separable ? 'text-term-green' : 'text-term-rose'}`}
            >
              {separable ? 'separable · a line divides the classes' : 'still entangled'}
            </span>
          </div>
          <div ref={rBox} className="flex-1 min-h-0 relative">
            <canvas ref={rCanvas} className="w-full h-full block" />
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
