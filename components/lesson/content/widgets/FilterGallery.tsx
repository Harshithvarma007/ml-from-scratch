'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

type KernelName = 'identity' | 'edge-sobel-x' | 'edge-sobel-y' | 'blur' | 'sharpen' | 'emboss'

const KERNELS: Record<KernelName, { k: number[][]; description: string }> = {
  identity: { k: [[0, 0, 0], [0, 1, 0], [0, 0, 0]], description: 'Pass-through. The trivial kernel.' },
  'edge-sobel-x': { k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], description: 'Vertical edges: positive where intensity increases left-to-right.' },
  'edge-sobel-y': { k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], description: 'Horizontal edges: positive where intensity increases top-to-bottom.' },
  blur: { k: Array(3).fill(0).map(() => Array(3).fill(1/9)), description: 'Box filter — averages a 3×3 neighborhood.' },
  sharpen: { k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], description: 'Enhances edges by subtracting neighbors from center.' },
  emboss: { k: [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], description: 'Directional emboss — highlights top-left-to-bottom-right gradient.' },
}

const SIZE = 20

// Hand-crafted "image" — a mix of shapes to show off each filter
function makeImage(): number[][] {
  const img: number[][] = []
  for (let y = 0; y < SIZE; y++) {
    const row: number[] = []
    for (let x = 0; x < SIZE; x++) {
      // Circle in the middle
      const dx = x - 10, dy = y - 10
      const r2 = dx * dx + dy * dy
      let v = 0
      if (r2 < 25) v = 1
      else if (r2 < 36) v = 0.5
      // Gradient band at top
      if (y < 4) v = Math.max(v, x / SIZE)
      row.push(v)
    }
    img.push(row)
  }
  return img
}

function convolve2D(img: number[][], kernel: number[][]): number[][] {
  const H = img.length, W = img[0].length
  const out: number[][] = []
  for (let y = 0; y < H; y++) {
    const row: number[] = []
    for (let x = 0; x < W; x++) {
      let s = 0
      for (let ky = 0; ky < 3; ky++) {
        for (let kx = 0; kx < 3; kx++) {
          const iy = y + ky - 1, ix = x + kx - 1
          const v = iy >= 0 && iy < H && ix >= 0 && ix < W ? img[iy][ix] : 0
          s += v * kernel[ky][kx]
        }
      }
      row.push(s)
    }
    out.push(row)
  }
  return out
}

export default function FilterGallery() {
  const [active, setActive] = useState<KernelName>('edge-sobel-x')
  const img = useMemo(() => makeImage(), [])
  const output = useMemo(() => convolve2D(img, KERNELS[active].k), [img, active])

  const absMax = Math.max(...output.flat().map(Math.abs), 1)

  const cell = (v: number, signed: boolean = true) => {
    const mag = Math.min(1, Math.abs(v) / absMax)
    const color = signed && v < 0 ? `rgba(244, 114, 182, ${mag})` : `rgba(251, 191, 36, ${mag})`
    return color
  }

  return (
    <WidgetFrame
      widgetName="FilterGallery"
      label="filter gallery — how a single kernel changes the world"
      right={<span className="font-mono">{KERNELS[active].description}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1 flex-wrap">
            {(Object.keys(KERNELS) as KernelName[]).map((k) => (
              <button
                key={k}
                onClick={() => setActive(k)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
                  active === k ? 'bg-dark-accent text-white' : 'border border-dark-border text-dark-text-secondary'
                )}
              >
                {k}
              </button>
            ))}
          </div>
          <Readout label="output range" value={`${Math.min(...output.flat()).toFixed(1)} … ${Math.max(...output.flat()).toFixed(1)}`} />
        </div>
      }
    >
      <div className="absolute inset-0 p-5 flex items-center justify-center gap-8 overflow-auto">
        {/* Input image */}
        <div className="flex flex-col items-center gap-2">
          <div className="text-[10px] font-mono text-dark-text-disabled uppercase tracking-wider">input</div>
          <div className="inline-grid" style={{ gridTemplateColumns: `repeat(${SIZE}, 12px)`, gap: 1 }}>
            {img.flatMap((row, y) => row.map((v, x) => (
              <div key={`${y}-${x}`} style={{ width: 12, height: 12, backgroundColor: `rgba(255, 255, 255, ${v})` }} />
            )))}
          </div>
        </div>

        {/* Kernel */}
        <div className="flex flex-col items-center gap-2">
          <div className="text-[10px] font-mono text-dark-text-disabled uppercase tracking-wider">kernel</div>
          <div className="inline-flex flex-col gap-1">
            {KERNELS[active].k.map((row, y) => (
              <div key={y} className="flex gap-1">
                {row.map((v, x) => (
                  <div key={x} className="w-10 h-10 rounded text-[10px] font-mono flex items-center justify-center bg-term-purple/30 text-term-purple tabular-nums">
                    {v.toFixed(2)}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* Output */}
        <div className="flex flex-col items-center gap-2">
          <div className="text-[10px] font-mono text-dark-text-disabled uppercase tracking-wider">output</div>
          <div className="inline-grid" style={{ gridTemplateColumns: `repeat(${SIZE}, 12px)`, gap: 1 }}>
            {output.flatMap((row, y) => row.map((v, x) => (
              <div key={`${y}-${x}`} style={{ width: 12, height: 12, backgroundColor: cell(v) }} />
            )))}
          </div>
          <div className="text-[10px] font-mono text-dark-text-disabled flex items-center gap-3">
            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-term-pink" /> negative</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-term-amber" /> positive</span>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
