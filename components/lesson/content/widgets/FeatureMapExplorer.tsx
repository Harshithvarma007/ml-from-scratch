'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Shows "feature maps" at 3 conv-layer depths for a simple synthetic input.
// Layer 1 = edges (Sobel-x, Sobel-y, Laplacian, diagonal). Layer 2 = corners
// and lines. Layer 3 = shape-like blobs. Pick a layer, see the grid of maps.

const SIZE = 16

function mulberry32(seed: number) {
  let t = seed >>> 0
  return () => {
    t += 0x6d2b79f5
    let x = Math.imul(t ^ (t >>> 15), t | 1)
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61)
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296
  }
}

type ImageName = 'circle' | 'cross' | 'square' | 'diagonal'

function makeImage(name: ImageName): number[][] {
  const img: number[][] = Array.from({ length: SIZE }, () => new Array(SIZE).fill(0))
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const cx = x - SIZE / 2
      const cy = y - SIZE / 2
      if (name === 'circle') img[y][x] = cx * cx + cy * cy < 25 ? 1 : 0
      if (name === 'cross') img[y][x] = Math.abs(cx) < 1.5 || Math.abs(cy) < 1.5 ? 1 : 0
      if (name === 'square') img[y][x] = Math.abs(cx) < 5 && Math.abs(cy) < 5 && (Math.abs(cx) > 3 || Math.abs(cy) > 3) ? 1 : 0
      if (name === 'diagonal') img[y][x] = Math.abs(cx - cy) < 1.5 ? 1 : 0
    }
  }
  return img
}

function conv2d(img: number[][], kernel: number[][]): number[][] {
  const H = img.length
  const W = img[0].length
  const k = kernel.length
  const half = Math.floor(k / 2)
  const out = Array.from({ length: H }, () => new Array(W).fill(0))
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let s = 0
      for (let ky = 0; ky < k; ky++) {
        for (let kx = 0; kx < k; kx++) {
          const iy = y + ky - half
          const ix = x + kx - half
          const v = iy >= 0 && iy < H && ix >= 0 && ix < W ? img[iy][ix] : 0
          s += v * kernel[ky][kx]
        }
      }
      out[y][x] = s
    }
  }
  return out
}

function relu(m: number[][]): number[][] {
  return m.map((r) => r.map((v) => Math.max(0, v)))
}

function pool2x2(m: number[][]): number[][] {
  const out: number[][] = []
  for (let y = 0; y < m.length; y += 2) {
    const row: number[] = []
    for (let x = 0; x < m[0].length; x += 2) {
      row.push(Math.max(m[y][x], m[y][x + 1] ?? 0, m[y + 1]?.[x] ?? 0, m[y + 1]?.[x + 1] ?? 0))
    }
    out.push(row)
  }
  return out
}

const L1_KERNELS = [
  { name: 'edge-x', k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] },
  { name: 'edge-y', k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] },
  { name: 'laplace', k: [[0, 1, 0], [1, -4, 1], [0, 1, 0]] },
  { name: 'diag', k: [[2, 1, 0], [1, 0, -1], [0, -1, -2]] },
]

const L2_KERNELS = [
  { name: 'corner', k: [[1, 0, -1], [0, 0, 0], [-1, 0, 1]] },
  { name: 'h-line', k: [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]] },
  { name: 'v-line', k: [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]] },
  { name: 'spot', k: [[1, 1, 1], [1, -8, 1], [1, 1, 1]] },
]

const L3_KERNELS = [
  { name: 'blob-A', k: [[1, 1, 1], [1, 1, 1], [1, 1, 1]] },
  { name: 'blob-B', k: [[0, 1, 0], [1, 2, 1], [0, 1, 0]] },
  { name: 'ring', k: [[1, 1, 1], [1, -6, 1], [1, 1, 1]] },
  { name: 'noise', k: (() => { const rng = mulberry32(5); return [[rng(), rng(), rng()], [rng(), rng(), rng()], [rng(), rng(), rng()]].map(r => r.map(v => v - 0.5)); })() },
]

export default function FeatureMapExplorer() {
  const [image, setImage] = useState<ImageName>('circle')
  const [layer, setLayer] = useState(1)

  const { maps, layerName } = useMemo(() => {
    const src = makeImage(image)
    if (layer === 0) return { maps: [{ name: 'input', m: src }], layerName: 'input' }
    const l1 = L1_KERNELS.map((k) => ({ name: k.name, m: relu(conv2d(src, k.k)) }))
    if (layer === 1) return { maps: l1, layerName: 'layer 1 (edge detectors)' }
    const pooled1 = l1.map((f) => ({ name: f.name, m: pool2x2(f.m) }))
    const l2 = L2_KERNELS.map((k, i) => ({ name: k.name, m: relu(conv2d(pooled1[i % pooled1.length].m, k.k)) }))
    if (layer === 2) return { maps: l2, layerName: 'layer 2 (composite detectors)' }
    const pooled2 = l2.map((f) => ({ name: f.name, m: pool2x2(f.m) }))
    const l3 = L3_KERNELS.map((k, i) => ({ name: k.name, m: relu(conv2d(pooled2[i % pooled2.length].m, k.k)) }))
    return { maps: l3, layerName: 'layer 3 (part / shape detectors)' }
  }, [image, layer])

  const absMax = Math.max(...maps.flatMap((f) => f.m.flat()).map(Math.abs), 1)

  return (
    <WidgetFrame
      widgetName="FeatureMapExplorer"
      label="feature maps — what each conv layer responds to"
      right={<span className="font-mono">synthetic input · ReLU + 2×2 max-pool between stages</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            {(['circle', 'cross', 'square', 'diagonal'] as ImageName[]).map((n) => (
              <button
                key={n}
                onClick={() => setImage(n)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono uppercase transition-all',
                  image === n
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary',
                )}
              >
                {n}
              </button>
            ))}
          </div>
          <Slider
            label="layer"
            value={layer}
            min={0}
            max={3}
            step={1}
            onChange={(v) => setLayer(Math.round(v))}
            format={(v) => `L${Math.round(v)}`}
            accent="accent-term-purple"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="current" value={layerName} accent="text-term-purple" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 flex items-center justify-center overflow-auto">
        <div className="flex items-start gap-4 flex-wrap justify-center">
          {maps.map((f, i) => {
            const H = f.m.length
            const W = f.m[0].length
            return (
              <div key={i} className="flex flex-col items-center gap-1">
                <div className="text-[10px] font-mono text-dark-text-disabled">{f.name}</div>
                <div
                  className="inline-grid"
                  style={{ gridTemplateColumns: `repeat(${W}, ${Math.max(6, Math.min(14, 180 / W))}px)`, gap: 1 }}
                >
                  {f.m.flatMap((row, y) =>
                    row.map((v, x) => {
                      const mag = Math.abs(v) / absMax
                      const color =
                        v >= 0
                          ? `rgba(251, 191, 36, ${Math.min(1, mag)})`
                          : `rgba(244, 114, 182, ${Math.min(1, mag)})`
                      return (
                        <div
                          key={`${y}-${x}`}
                          style={{
                            backgroundColor: color,
                            width: `${Math.max(6, Math.min(14, 180 / W))}px`,
                            height: `${Math.max(6, Math.min(14, 180 / W))}px`,
                          }}
                        />
                      )
                    }),
                  )}
                </div>
                <div className="text-[9px] font-mono text-dark-text-disabled tabular-nums">
                  {H}×{W}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </WidgetFrame>
  )
}
