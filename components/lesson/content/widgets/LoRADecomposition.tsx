'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Visualize the LoRA decomposition W = W_0 + B·A. W_0 (d × d) is drawn as a
// big frozen grid. A (r × d) and B (d × r) are tall/wide skinny rectangles.
// A slider for r shows how the product B·A reconstructs a d × d update on
// the left — but is parameterized by only 2·r·d trainable weights. Numeric
// readouts compare 2·r·d to d² so the savings are concrete.

const D_DISPLAY = 16 // grid cells per side of the visual matrix
const D_REAL = 4096   // "real" dimension for parameter counting

function hashCell(i: number, j: number, seed: number): number {
  const h = Math.sin(i * 37 + j * 17 + seed * 91) * 43758.5453
  return h - Math.floor(h)
}

// Mini BA product visualization: build a seeded rank-r update.
function buildLowRankUpdate(r: number): number[][] {
  const A: number[][] = []
  const B: number[][] = []
  for (let k = 0; k < r; k++) {
    const arow: number[] = []
    for (let j = 0; j < D_DISPLAY; j++) arow.push((hashCell(k, j, 7) - 0.5) * 0.8)
    A.push(arow)
  }
  for (let i = 0; i < D_DISPLAY; i++) {
    const brow: number[] = []
    for (let k = 0; k < r; k++) brow.push((hashCell(i, k, 23) - 0.5) * 0.8)
    B.push(brow)
  }
  const U: number[][] = []
  for (let i = 0; i < D_DISPLAY; i++) {
    const row: number[] = []
    for (let j = 0; j < D_DISPLAY; j++) {
      let s = 0
      for (let k = 0; k < r; k++) s += B[i][k] * A[k][j]
      row.push(s)
    }
    U.push(row)
  }
  return U
}

export default function LoRADecomposition() {
  const [r, setR] = useState(8)

  const update = useMemo(() => buildLowRankUpdate(r), [r])

  const fullParams = D_REAL * D_REAL
  const loraParams = 2 * r * D_REAL
  const ratio = loraParams / fullParams
  const savings = 1 - ratio

  return (
    <WidgetFrame
      widgetName="LoRADecomposition"
      label="LoRA decomposition — W = W_0 + B·A"
      right={<span className="font-mono">d = {D_REAL.toLocaleString('en-US')} · trainable only in B and A</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="rank r"
            value={r}
            min={1}
            max={64}
            step={1}
            onChange={(v) => setR(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-cyan"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="trainable" value={loraParams.toLocaleString('en-US')} accent="text-term-green" />
            <Readout label="full W" value={fullParams.toLocaleString('en-US')} accent="text-dark-text-secondary" />
            <Readout label="ratio" value={`${(ratio * 100).toFixed(2)}%`} accent="text-term-amber" />
            <Readout label="savings" value={`${(savings * 100).toFixed(2)}%`} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <div className="h-full grid grid-cols-[1fr_auto_1fr_auto_auto_auto_auto] items-center gap-3 font-mono text-[10.5px]">
          {/* W_0 */}
          <div className="flex flex-col items-center gap-1.5 min-w-0">
            <div className="text-dark-text-disabled uppercase tracking-wider text-[9.5px]">W_0 (frozen)</div>
            <MatrixGrid rows={D_DISPLAY} cols={D_DISPLAY} color="gray" fill={(i, j) => hashCell(i, j, 1) * 0.4 + 0.1} />
            <div className="text-dark-text-muted">d × d = {fullParams.toLocaleString('en-US')}</div>
          </div>
          <div className="text-dark-text-secondary text-[18px] font-light self-center">+</div>

          {/* B·A product */}
          <div className="flex flex-col items-center gap-1.5 min-w-0">
            <div className="text-term-cyan uppercase tracking-wider text-[9.5px]">B · A (update)</div>
            <MatrixGrid rows={D_DISPLAY} cols={D_DISPLAY} color="cyan" fill={(i, j) => update[i][j] * 1.2 + 0.5} bipolar values={update} />
            <div className="text-term-cyan">rank r = {r}</div>
          </div>

          <div className="text-dark-text-secondary text-[14px] self-center">=</div>

          {/* B matrix */}
          <div className="flex flex-col items-center gap-1.5">
            <div className="text-term-amber uppercase tracking-wider text-[9.5px]">B</div>
            <MatrixGrid rows={D_DISPLAY} cols={Math.max(1, Math.min(r, 12))} color="amber" fill={(i, j) => hashCell(i, j, 23) * 0.5 + 0.3} cellPx={10} />
            <div className="text-dark-text-muted">d × r</div>
          </div>

          <div className="text-dark-text-secondary text-[14px] self-center">·</div>

          {/* A matrix */}
          <div className="flex flex-col items-center gap-1.5">
            <div className="text-term-pink uppercase tracking-wider text-[9.5px]">A</div>
            <MatrixGrid rows={Math.max(1, Math.min(r, 12))} cols={D_DISPLAY} color="pink" fill={(i, j) => hashCell(i, j, 7) * 0.5 + 0.3} cellPx={10} />
            <div className="text-dark-text-muted">r × d</div>
          </div>

          <div className="col-span-7 flex flex-col gap-1 pt-2">
            <div className="flex items-center justify-between text-dark-text-disabled uppercase tracking-wider text-[9.5px]">
              <span>parameter budget</span>
              <span>2·r·d vs d²</span>
            </div>
            <div className="relative h-3 bg-dark-surface-elevated/40 rounded overflow-hidden">
              <div className="absolute inset-y-0 left-0 bg-dark-text-disabled/40" style={{ width: '100%' }} />
              <div className="absolute inset-y-0 left-0 bg-term-green" style={{ width: `${Math.max(ratio * 100, 0.4)}%` }} />
            </div>
            <div className="flex items-center justify-between font-mono text-[10px] text-dark-text-muted">
              <span>
                <span className="text-term-green">trainable:</span> {loraParams.toLocaleString('en-US')}
              </span>
              <span>
                <span className="text-dark-text-secondary">full W:</span> {fullParams.toLocaleString('en-US')}
              </span>
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function MatrixGrid({
  rows,
  cols,
  fill,
  color,
  cellPx,
  bipolar,
  values,
}: {
  rows: number
  cols: number
  fill: (i: number, j: number) => number
  color: 'gray' | 'cyan' | 'amber' | 'pink'
  cellPx?: number
  bipolar?: boolean
  values?: number[][]
}) {
  const base = {
    gray: [120, 120, 125],
    cyan: [103, 232, 249],
    amber: [251, 191, 36],
    pink: [244, 114, 182],
  }[color]
  const neg = [103, 232, 249]
  const size = cellPx ?? 12
  return (
    <div
      className="grid gap-[1px] p-1 rounded bg-dark-bg border border-dark-border"
      style={{
        gridTemplateColumns: `repeat(${cols}, ${size}px)`,
        gridTemplateRows: `repeat(${rows}, ${size}px)`,
      }}
    >
      {Array.from({ length: rows }).flatMap((_, i) =>
        Array.from({ length: cols }).map((_, j) => {
          const v = values ? values[i][j] : fill(i, j)
          const mag = Math.max(0, Math.min(1, Math.abs(v)))
          const rgb = bipolar && v < 0 ? neg : base
          const alpha = 0.12 + mag * 0.8
          return (
            <div
              key={`${i}-${j}`}
              className="rounded-[1px]"
              style={{ backgroundColor: `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})` }}
            />
          )
        }),
      )}
    </div>
  )
}
