'use client'

import { useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Visualize how a d_model = 64 hidden vector gets sliced into `num_heads`
// independent head-dim chunks. The top bar is one 64-d vector, the dividers
// show where each head starts. Below: param counts and a warning when the
// head_dim gets so small that each head can't carry enough information.

const D_MODEL = 64
const HEAD_OPTIONS = [1, 2, 4, 8, 16] as const

// Deterministic "activation values" per dimension so the bar actually looks
// like data instead of a uniform stripe.
function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function buildActivations(): number[] {
  const rng = mulberry32(7)
  const out: number[] = []
  for (let i = 0; i < D_MODEL; i++) {
    // mix of slow wave and noise — looks organic
    const wave = Math.sin((i / D_MODEL) * Math.PI * 2.4) * 0.5
    out.push(wave + (rng() * 2 - 1) * 0.35)
  }
  return out
}

const ACTIVATIONS = buildActivations()

const HEAD_COLORS = [
  '#f87171', '#fbbf24', '#4ade80', '#67e8f9',
  '#a78bfa', '#f472b6', '#fb923c', '#38bdf8',
  '#facc15', '#34d399', '#60a5fa', '#c084fc',
  '#ec4899', '#fcd34d', '#86efac', '#93c5fd',
]

export default function MultiHeadSplit() {
  const [numHeads, setNumHeads] = useState<number>(8)
  const headDim = D_MODEL / numHeads

  // Per-head projection params: head_dim × d_model × 3 (Q, K, V)
  const paramsPerHead = headDim * D_MODEL * 3
  const paramsTotalQKV = paramsPerHead * numHeads
  const paramsOutProj = D_MODEL * D_MODEL
  const paramsTotal = paramsTotalQKV + paramsOutProj

  const warning = headDim < 4
  const sweet = headDim >= 16 && numHeads >= 4

  return (
    <WidgetFrame
      widgetName="MultiHeadSplit"
      label="multi-head split — carve one vector into many heads"
      right={<span className="font-mono">d_model = {D_MODEL} · heads = {numHeads} · head_dim = {headDim}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mr-1">heads</span>
            {HEAD_OPTIONS.map((h) => (
              <button
                key={h}
                onClick={() => setNumHeads(h)}
                className={cn(
                  'w-10 py-1 rounded text-[11px] font-mono transition-all',
                  numHeads === h
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {h}
              </button>
            ))}
          </div>
          <div className="ml-auto flex items-center gap-4">
            <Readout label="head_dim" value={String(headDim)} accent={warning ? 'text-term-rose' : 'text-term-cyan'} />
            <Readout label="params (QKV+O)" value={paramsTotal.toLocaleString()} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden flex flex-col gap-4">
        {/* Top: the 64-d vector as a single horizontal bar */}
        <div>
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-1.5">
            one hidden vector — {D_MODEL} dimensions
          </div>
          <div className="relative h-12 bg-dark-bg rounded overflow-hidden flex">
            {ACTIVATIONS.map((v, i) => {
              const headIdx = Math.floor(i / headDim)
              const color = HEAD_COLORS[headIdx % HEAD_COLORS.length]
              const intensity = 0.25 + Math.min(1, Math.abs(v)) * 0.7
              return (
                <div
                  key={i}
                  className="flex-1 transition-all"
                  style={{
                    backgroundColor: color,
                    opacity: intensity,
                  }}
                  title={`dim ${i} = ${v.toFixed(3)} (head ${headIdx})`}
                />
              )
            })}
            {/* Divider lines between heads */}
            {Array.from({ length: numHeads - 1 }).map((_, k) => (
              <div
                key={k}
                className="absolute top-0 bottom-0 border-l-2 border-dark-bg"
                style={{ left: `${((k + 1) / numHeads) * 100}%` }}
              />
            ))}
          </div>
          {/* Index ticks */}
          <div className="flex mt-1 font-mono text-[9px] text-dark-text-disabled">
            {Array.from({ length: numHeads }).map((_, h) => (
              <div
                key={h}
                className="text-center border-l border-dark-border first:border-l-0 py-0.5"
                style={{ width: `${100 / numHeads}%`, color: HEAD_COLORS[h % HEAD_COLORS.length] }}
              >
                head {h}
              </div>
            ))}
          </div>
        </div>

        {/* Per-head breakdown as stacked rows */}
        <div className="flex-1 min-h-0 overflow-auto">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-1.5">
            each head operates on its own slice — {headDim} dims
          </div>
          <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${Math.min(numHeads, 4)}, minmax(0, 1fr))` }}>
            {Array.from({ length: numHeads }).map((_, h) => {
              const slice = ACTIVATIONS.slice(h * headDim, (h + 1) * headDim)
              const color = HEAD_COLORS[h % HEAD_COLORS.length]
              return (
                <div
                  key={h}
                  className="rounded border border-dark-border p-2"
                  style={{ backgroundColor: `${color}10` }}
                >
                  <div className="flex items-center justify-between text-[10px] font-mono mb-1">
                    <span style={{ color }}>head {h}</span>
                    <span className="text-dark-text-disabled">[{h * headDim}..{(h + 1) * headDim - 1}]</span>
                  </div>
                  <div className="flex gap-[1px] h-4">
                    {slice.map((v, i) => (
                      <div
                        key={i}
                        className="flex-1 rounded-[1px]"
                        style={{
                          backgroundColor: color,
                          opacity: 0.25 + Math.min(1, Math.abs(v)) * 0.7,
                        }}
                      />
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Parameter readout + warnings */}
        <div className="grid grid-cols-[1fr_auto] items-end gap-4 text-[11px] font-mono">
          <div className="flex flex-col gap-0.5">
            <span className="text-dark-text-muted">
              <span className="text-dark-text-disabled">per head:</span>{' '}
              <span className="text-term-cyan">{headDim}</span> × <span className="text-term-cyan">{D_MODEL}</span> × 3 (QKV) ={' '}
              <span className="text-term-amber">{paramsPerHead.toLocaleString()}</span>
            </span>
            <span className="text-dark-text-muted">
              <span className="text-dark-text-disabled">× {numHeads} heads + W_O ({D_MODEL}²) =</span>{' '}
              <span className="text-term-amber">{paramsTotal.toLocaleString()}</span> total
            </span>
            <span className="text-dark-text-disabled italic text-[10.5px]">
              note: QKV total is invariant to num_heads — heads are a view, not a cost.
            </span>
          </div>
          <div className="text-right">
            {warning && (
              <div className="text-term-rose text-[11px]">
                warning: head_dim = {headDim} is too small. Each head can barely
                <br />represent anything — pick fewer, wider heads.
              </div>
            )}
            {sweet && !warning && (
              <div className="text-term-green text-[11px]">sweet spot: heads wide enough to specialize.</div>
            )}
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
