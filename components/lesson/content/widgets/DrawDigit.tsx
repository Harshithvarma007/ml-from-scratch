'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Button, Readout, Slider } from './WidgetFrame'
import { RotateCcw, Eraser } from 'lucide-react'
import { cn } from '@/lib/utils'

// A small canvas you can draw on. We rasterize to a 14×14 grid, compare
// against 10 hand-crafted digit templates via cosine similarity, softmax the
// scores, and display the predicted distribution. Not a real MNIST model, but
// it captures the "see your digit turn into a distribution over classes" feel.

const CANVAS_SIZE = 224
const GRID_SIZE = 14

// 14×14 hand-drawn binary templates for digits 0-9. Each string is one row,
// '#' = active, '.' = inactive. They're rough but recognisable.
const TEMPLATES: string[][] = [
  // 0
  [
    '..............',
    '....######....',
    '...##....##...',
    '..##......##..',
    '..##......##..',
    '..##......##..',
    '..##......##..',
    '..##......##..',
    '..##......##..',
    '..##......##..',
    '...##....##...',
    '....######....',
    '..............',
    '..............',
  ],
  // 1
  [
    '..............',
    '......###.....',
    '.....####.....',
    '....#.##......',
    '......##......',
    '......##......',
    '......##......',
    '......##......',
    '......##......',
    '......##......',
    '......##......',
    '....######....',
    '..............',
    '..............',
  ],
  // 2
  [
    '..............',
    '....#####.....',
    '...##...##....',
    '........##....',
    '........##....',
    '.......##.....',
    '......##......',
    '.....##.......',
    '....##........',
    '...##.........',
    '..##..........',
    '..##########..',
    '..............',
    '..............',
  ],
  // 3
  [
    '..............',
    '...######.....',
    '..........##..',
    '..........##..',
    '..........##..',
    '....######....',
    '..........##..',
    '..........##..',
    '..........##..',
    '..........##..',
    '..........##..',
    '...######.....',
    '..............',
    '..............',
  ],
  // 4
  [
    '..............',
    '........##....',
    '.......###....',
    '......####....',
    '.....##.##....',
    '....##..##....',
    '...##...##....',
    '..##....##....',
    '..##########..',
    '........##....',
    '........##....',
    '........##....',
    '..............',
    '..............',
  ],
  // 5
  [
    '..............',
    '..##########..',
    '..##..........',
    '..##..........',
    '..##..........',
    '..########....',
    '..........##..',
    '..........##..',
    '..........##..',
    '..........##..',
    '..##....##....',
    '...######.....',
    '..............',
    '..............',
  ],
  // 6
  [
    '..............',
    '......#####...',
    '....##........',
    '...##.........',
    '..##..........',
    '..##########..',
    '..##......##..',
    '..##......##..',
    '..##......##..',
    '..##......##..',
    '..##......##..',
    '...######.....',
    '..............',
    '..............',
  ],
  // 7
  [
    '..............',
    '..##########..',
    '..........##..',
    '.........##...',
    '........##....',
    '.......##.....',
    '......##......',
    '.....##.......',
    '....##........',
    '...##.........',
    '...##.........',
    '...##.........',
    '..............',
    '..............',
  ],
  // 8
  [
    '..............',
    '....######....',
    '...##....##...',
    '..##......##..',
    '..##......##..',
    '...##....##...',
    '....######....',
    '...##....##...',
    '..##......##..',
    '..##......##..',
    '...##....##...',
    '....######....',
    '..............',
    '..............',
  ],
  // 9
  [
    '..............',
    '....######....',
    '...##....##...',
    '..##......##..',
    '..##......##..',
    '..##......##..',
    '...########...',
    '..........##..',
    '..........##..',
    '.........##...',
    '....##..##....',
    '....####......',
    '..............',
    '..............',
  ],
]

const TEMPLATE_VECTORS: number[][] = TEMPLATES.map((rows) =>
  rows.flatMap((row) => row.split('').map((c) => (c === '#' ? 1 : 0))),
)

function softmax(z: number[], temperature = 0.5): number[] {
  const scaled = z.map((v) => v / temperature)
  const m = Math.max(...scaled)
  const exps = scaled.map((v) => Math.exp(v - m))
  const sum = exps.reduce((a, b) => a + b, 0) || 1
  return exps.map((e) => e / sum)
}

function cosineSim(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const denom = Math.sqrt(na * nb) || 1
  return dot / denom
}

export default function DrawDigit() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const drawingRef = useRef<boolean>(false)
  const [grid, setGrid] = useState<number[]>(new Array(GRID_SIZE * GRID_SIZE).fill(0))
  const [brush, setBrush] = useState(14)

  const probs = useMemo(() => {
    // If empty, flat distribution
    if (grid.every((v) => v === 0)) return new Array(10).fill(0.1)
    const sims = TEMPLATE_VECTORS.map((t) => cosineSim(grid, t))
    return softmax(sims, 0.1)
  }, [grid])

  const top1 = probs.indexOf(Math.max(...probs))

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = CANVAS_SIZE * dpr
    canvas.height = CANVAS_SIZE * dpr
    canvas.style.width = `${CANVAS_SIZE}px`
    canvas.style.height = `${CANVAS_SIZE}px`
    const ctx = canvas.getContext('2d')!
    ctx.scale(dpr, dpr)
    ctx.fillStyle = '#0a0a0a'
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
  }, [])

  const paintAt = (e: React.PointerEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * CANVAS_SIZE
    const y = ((e.clientY - rect.top) / rect.height) * CANVAS_SIZE
    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = '#fbbf24'
    ctx.beginPath()
    ctx.arc(x, y, brush, 0, Math.PI * 2)
    ctx.fill()
    // Also update our coarse 14×14 grid by stamping ones nearby
    const cellSize = CANVAS_SIZE / GRID_SIZE
    const gx = Math.floor(x / cellSize)
    const gy = Math.floor(y / cellSize)
    const R = Math.max(1, Math.round(brush / cellSize))
    setGrid((prev) => {
      const next = [...prev]
      for (let dy = -R; dy <= R; dy++) {
        for (let dx = -R; dx <= R; dx++) {
          const nx = gx + dx, ny = gy + dy
          if (nx < 0 || ny < 0 || nx >= GRID_SIZE || ny >= GRID_SIZE) continue
          if (dx * dx + dy * dy > R * R) continue
          next[ny * GRID_SIZE + nx] = 1
        }
      }
      return next
    })
  }

  const clear = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = '#0a0a0a'
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
    setGrid(new Array(GRID_SIZE * GRID_SIZE).fill(0))
  }

  return (
    <WidgetFrame
      widgetName="DrawDigit"
      label="draw a digit — watch the probability distribution update"
      right={<span className="font-mono">template-matching classifier · not a real NN</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider label="brush" value={brush} min={6} max={24} step={1} onChange={setBrush} accent="accent-term-amber" />
          <Button onClick={clear}>
            <Eraser className="w-3 h-3 inline -mt-px mr-1" /> clear
          </Button>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="top-1" value={String(top1)} accent="text-term-amber" />
            <Readout label="confidence" value={`${(probs[top1] * 100).toFixed(0)}%`} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 grid grid-cols-[auto_1fr] gap-6 items-center justify-items-center">
        {/* Canvas */}
        <div className="flex flex-col items-center gap-2">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            draw here
          </div>
          <canvas
            ref={canvasRef}
            className="rounded border-2 border-dark-border cursor-crosshair bg-dark-bg touch-none"
            onPointerDown={(e) => {
              drawingRef.current = true
              paintAt(e)
            }}
            onPointerMove={(e) => {
              if (drawingRef.current) paintAt(e)
            }}
            onPointerUp={() => (drawingRef.current = false)}
            onPointerLeave={() => (drawingRef.current = false)}
          />
        </div>

        {/* Probability bars */}
        <div className="w-full max-w-[460px]">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
            predicted probability
          </div>
          <div className="space-y-1.5">
            {probs.map((p, i) => (
              <div key={i} className="flex items-center gap-3 font-mono text-[11px]">
                <span className={cn('w-5 text-right', i === top1 ? 'text-term-amber font-semibold' : 'text-dark-text-muted')}>
                  {i}
                </span>
                <div className="flex-1 h-5 bg-dark-surface-elevated/40 rounded overflow-hidden">
                  <div
                    className={cn('h-full transition-all', i === top1 ? 'bg-term-amber/70' : 'bg-term-amber/30')}
                    style={{ width: `${(p * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className={cn('w-12 text-right tabular-nums', i === top1 ? 'text-term-amber' : 'text-dark-text-muted')}>
                  {(p * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
