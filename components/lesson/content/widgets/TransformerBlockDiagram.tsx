'use client'

import { useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Interactive SVG of one transformer block. Nodes are clickable — picking one
// highlights the path and fills the side panel with that node's role + its
// FLOP cost under the current (d_model, seq_len). Sliders let you switch
// between small/medium/large scales and watch where the cost goes.

type NodeId =
  | 'input'
  | 'ln1'
  | 'mha'
  | 'add1'
  | 'ln2'
  | 'mlp'
  | 'add2'
  | 'output'

type NodeInfo = {
  id: NodeId
  x: number
  y: number
  w: number
  h: number
  label: string
  color: string
  role: string
  /** FLOPs given (d, T) — approximate; residuals ignore bias */
  flops: (d: number, T: number) => number
}

const NODES: NodeInfo[] = [
  {
    id: 'input', x: 50, y: 50, w: 160, h: 40, label: 'input x', color: '#94a3b8',
    role: 'input sequence of shape (T, d_model). Starts the block.',
    flops: () => 0,
  },
  {
    id: 'ln1', x: 50, y: 120, w: 160, h: 40, label: 'LayerNorm', color: '#67e8f9',
    role: 'pre-norm: subtract mean, scale by rmsd, apply learned gain & bias per feature.',
    flops: (d, T) => 5 * d * T,
  },
  {
    id: 'mha', x: 50, y: 190, w: 160, h: 60, label: 'Multi-Head Attention', color: '#a78bfa',
    role: 'QKV projection, attention matmul, output projection. Cost scales with T·d + T²·d.',
    flops: (d, T) => 4 * d * d * T + 2 * d * T * T,
  },
  {
    id: 'add1', x: 50, y: 280, w: 160, h: 40, label: 'add residual', color: '#fbbf24',
    role: 'x = x + MHA(LN(x)). Keeps the gradient highway open across depth.',
    flops: (d, T) => d * T,
  },
  {
    id: 'ln2', x: 50, y: 350, w: 160, h: 40, label: 'LayerNorm', color: '#67e8f9',
    role: 'second normalization before the MLP sub-layer.',
    flops: (d, T) => 5 * d * T,
  },
  {
    id: 'mlp', x: 50, y: 420, w: 160, h: 60, label: 'MLP (4d up + down)', color: '#f472b6',
    role: 'two linear layers with GELU in between: d → 4d → d. Most of the params live here.',
    flops: (d, T) => 16 * d * d * T,
  },
  {
    id: 'add2', x: 50, y: 510, w: 160, h: 40, label: 'add residual', color: '#fbbf24',
    role: 'x = x + MLP(LN(x)). Block output is the sum of two residual streams.',
    flops: (d, T) => d * T,
  },
  {
    id: 'output', x: 50, y: 580, w: 160, h: 40, label: 'output y', color: '#4ade80',
    role: 'shape (T, d_model). Feeds the next transformer block.',
    flops: () => 0,
  },
]

const D_OPTIONS = [256, 512, 1024] as const
const T_OPTIONS = [128, 512, 1024] as const

function formatFlops(n: number): string {
  if (n >= 1e12) return (n / 1e12).toFixed(2) + ' T'
  if (n >= 1e9) return (n / 1e9).toFixed(2) + ' G'
  if (n >= 1e6) return (n / 1e6).toFixed(2) + ' M'
  if (n >= 1e3) return (n / 1e3).toFixed(2) + ' K'
  return n.toFixed(0)
}

export default function TransformerBlockDiagram() {
  const [dModel, setDModel] = useState<number>(512)
  const [seqLen, setSeqLen] = useState<number>(512)
  const [selected, setSelected] = useState<NodeId>('mha')

  const totalFlops = NODES.reduce((a, n) => a + n.flops(dModel, seqLen), 0)
  const selectedNode = NODES.find((n) => n.id === selected)!
  const selectedFlops = selectedNode.flops(dModel, seqLen)
  const selectedPct = totalFlops > 0 ? (selectedFlops / totalFlops) * 100 : 0

  return (
    <WidgetFrame
      widgetName="TransformerBlockDiagram"
      label="transformer block — click any node"
      right={<span className="font-mono">d = {dModel} · T = {seqLen}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mr-1">d_model</span>
            {D_OPTIONS.map((d) => (
              <button
                key={d}
                onClick={() => setDModel(d)}
                className={cn(
                  'w-12 py-1 rounded text-[10.5px] font-mono transition-all',
                  dModel === d
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {d}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mr-1">seq_len</span>
            {T_OPTIONS.map((t) => (
              <button
                key={t}
                onClick={() => setSeqLen(t)}
                className={cn(
                  'w-12 py-1 rounded text-[10.5px] font-mono transition-all',
                  seqLen === t
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {t}
              </button>
            ))}
          </div>
          <div className="ml-auto flex items-center gap-4">
            <Readout label="block FLOPs" value={formatFlops(totalFlops)} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-1 md:grid-cols-[260px_1fr] gap-4">
        {/* SVG diagram */}
        <div className="relative flex items-center justify-center min-h-0 overflow-hidden">
          <svg viewBox="0 0 260 670" className="h-full w-full">
            {/* Wires */}
            {renderWires()}
            {/* Residual loops */}
            <path d="M 240 70 Q 260 70 260 300 Q 260 300 230 300" stroke="#fbbf24" strokeWidth="1.5" fill="none" strokeDasharray="4 3" opacity="0.7" />
            <path d="M 240 340 Q 258 340 258 530 Q 258 530 230 530" stroke="#fbbf24" strokeWidth="1.5" fill="none" strokeDasharray="4 3" opacity="0.7" />
            <text x="266" y="180" fontSize="9" fill="#fbbf24" fontFamily="JetBrains Mono, monospace" writingMode="vertical-rl">residual</text>
            <text x="266" y="420" fontSize="9" fill="#fbbf24" fontFamily="JetBrains Mono, monospace" writingMode="vertical-rl">residual</text>

            {/* Nodes */}
            {NODES.map((n) => {
              const flopShare = totalFlops > 0 ? n.flops(dModel, seqLen) / totalFlops : 0
              const isSel = selected === n.id
              return (
                <g
                  key={n.id}
                  onClick={() => setSelected(n.id)}
                  className="cursor-pointer"
                >
                  <rect
                    x={n.x}
                    y={n.y}
                    width={n.w}
                    height={n.h}
                    rx={8}
                    fill={isSel ? `${n.color}22` : '#0f1018'}
                    stroke={n.color}
                    strokeWidth={isSel ? 2 : 1.1}
                  />
                  {flopShare > 0 && (
                    <rect
                      x={n.x}
                      y={n.y + n.h - 4}
                      width={n.w * Math.min(1, flopShare)}
                      height={3}
                      fill={n.color}
                      opacity={0.55}
                    />
                  )}
                  <text
                    x={n.x + n.w / 2}
                    y={n.y + n.h / 2 + 4}
                    textAnchor="middle"
                    fontSize="11"
                    fontFamily="JetBrains Mono, monospace"
                    fill={isSel ? '#ffffff' : n.color}
                  >
                    {n.label}
                  </text>
                </g>
              )
            })}
            <defs>
              <marker id="arrow-block" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="#3f3f46" />
              </marker>
            </defs>
          </svg>
        </div>

        {/* Details panel */}
        <div className="flex flex-col gap-3 min-h-0">
          <div className="rounded border border-dark-border p-3" style={{ backgroundColor: `${selectedNode.color}10` }}>
            <div className="text-[10px] font-mono uppercase tracking-wider mb-1" style={{ color: selectedNode.color }}>
              {selectedNode.label}
            </div>
            <p className="font-mono text-[11.5px] text-dark-text-primary leading-relaxed">
              {selectedNode.role}
            </p>
          </div>

          <div className="rounded border border-dark-border p-3 bg-dark-surface-elevated/30">
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
              FLOPs at d = {dModel}, T = {seqLen}
            </div>
            <div className="flex items-baseline gap-3 mb-2">
              <span className="text-[18px] font-mono text-term-amber tabular-nums">
                {formatFlops(selectedFlops)}
              </span>
              <span className="text-[11px] font-mono text-dark-text-muted">
                {selectedPct.toFixed(1)}% of block
              </span>
            </div>
            <div className="h-2 bg-dark-bg rounded-full overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{ width: `${selectedPct}%`, backgroundColor: selectedNode.color }}
              />
            </div>
          </div>

          <div className="rounded border border-dark-border p-3">
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
              breakdown — per layer
            </div>
            <div className="flex flex-col gap-1">
              {NODES.filter((n) => n.flops(dModel, seqLen) > 0).map((n) => {
                const f = n.flops(dModel, seqLen)
                const pct = (f / totalFlops) * 100
                return (
                  <div key={n.id} className="flex items-center gap-2 font-mono text-[10.5px]">
                    <span className="w-[120px] truncate" style={{ color: n.color }}>{n.label}</span>
                    <div className="flex-1 h-2 bg-dark-surface-elevated/40 rounded-sm overflow-hidden">
                      <div className="h-full" style={{ width: `${pct}%`, backgroundColor: n.color, opacity: 0.75 }} />
                    </div>
                    <span className="w-20 text-right text-dark-text-primary tabular-nums">{formatFlops(f)}</span>
                    <span className="w-10 text-right text-dark-text-disabled tabular-nums">{pct.toFixed(0)}%</span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function renderWires() {
  // Connect each node's bottom to the next node's top
  const flow: NodeId[] = ['input', 'ln1', 'mha', 'add1', 'ln2', 'mlp', 'add2', 'output']
  const byId = Object.fromEntries(NODES.map((n) => [n.id, n]))
  return flow.slice(0, -1).map((id, i) => {
    const from = byId[id]
    const to = byId[flow[i + 1]]
    const x1 = from.x + from.w / 2
    const y1 = from.y + from.h
    const x2 = to.x + to.w / 2
    const y2 = to.y
    return (
      <line
        key={`wire-${i}`}
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke="#3f3f46"
        strokeWidth={1.2}
        markerEnd="url(#arrow-block)"
      />
    )
  })
}
