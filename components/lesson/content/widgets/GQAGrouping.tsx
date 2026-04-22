'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Visualize GQA head grouping. A row of Q heads at the top connects to a
// smaller row of KV heads at the bottom. Pick n_kv_heads from a preset list —
// kv=q is MHA, kv<q is GQA, kv=1 is MQA. Draw fan-in lines from each KV head
// to the group of Q heads that share it.

const N_Q = 32

const KV_OPTIONS = [1, 2, 4, 8, 16, 32]

function regimeFor(n_kv: number): { name: string; color: string; desc: string } {
  if (n_kv === N_Q) return { name: 'MHA (Multi-Head)', color: '#67e8f9', desc: 'each Q head owns its K/V — maximum parameters, maximum cache, the classic transformer.' }
  if (n_kv === 1) return { name: 'MQA (Multi-Query)', color: '#f472b6', desc: 'all Q heads share one K/V — smallest cache. faster, some quality loss on harder tasks.' }
  return { name: 'GQA (Grouped-Query)', color: '#fbbf24', desc: `${N_Q / n_kv} Q heads share each K/V — the Llama-2 70B compromise. MHA accuracy at MQA speed.` }
}

export default function GQAGrouping() {
  const [n_kv, setN_kv] = useState(8)
  const group = Math.floor(N_Q / n_kv)
  const regime = regimeFor(n_kv)

  const layout = useMemo(() => {
    const W = 920
    const padLR = 40
    const qSpacing = (W - 2 * padLR) / (N_Q - 1)
    const kvSpacing = n_kv > 1 ? (W - 2 * padLR) / (n_kv - 1) : 0
    const qY = 70
    const kvY = 280
    const qPositions = Array.from({ length: N_Q }, (_, i) => padLR + i * qSpacing)
    const kvPositions = Array.from({ length: n_kv }, (_, i) =>
      n_kv === 1 ? W / 2 : padLR + i * kvSpacing,
    )
    // assign q[i] -> kv[floor(i / group)]
    const assignments = qPositions.map((x, i) => ({
      qX: x,
      kvIdx: Math.min(n_kv - 1, Math.floor(i / group)),
    }))
    return { W, qPositions, kvPositions, assignments, qY, kvY }
  }, [n_kv, group])

  const cacheReductionVsMHA = n_kv / N_Q

  return (
    <WidgetFrame
      widgetName="GQAGrouping"
      label="GQA head grouping — which Q heads share which K/V"
      right={<span className="font-mono">n_q = {N_Q} · n_kv = {n_kv} · group = {group}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="font-mono text-[11px] text-dark-text-secondary">n_kv_heads</span>
            <div className="flex items-center gap-0.5">
              {KV_OPTIONS.map((v) => (
                <button
                  key={v}
                  onClick={() => setN_kv(v)}
                  className={cn(
                    'px-2 py-0.5 rounded text-[10.5px] font-mono transition-all',
                    n_kv === v
                      ? 'bg-dark-accent text-white'
                      : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                  )}
                >
                  {v}
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="regime" value={regime.name} accent="text-term-amber" />
            <Readout label="cache vs MHA" value={`${(cacheReductionVsMHA * 100).toFixed(0)}%`} accent="text-term-green" />
            <Readout label="queries / KV" value={String(group)} accent="text-term-cyan" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 flex flex-col gap-2 overflow-hidden">
        <div className="flex-1 min-h-0 relative">
          <svg viewBox={`0 0 ${layout.W} 340`} className="w-full h-full">
            <defs>
              <marker id="gqa-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="#525252" />
              </marker>
            </defs>

            {/* Q labels */}
            <text x={20} y={30} fontSize="11" fill="#fbbf24" fontFamily="JetBrains Mono, monospace">
              query heads · 1 → {N_Q}
            </text>

            {/* Connection lines */}
            {layout.assignments.map((a, i) => {
              const kvX = layout.kvPositions[a.kvIdx]
              return (
                <line
                  key={`line-${i}`}
                  x1={a.qX}
                  y1={layout.qY + 10}
                  x2={kvX}
                  y2={layout.kvY - 10}
                  stroke={regime.color}
                  strokeWidth={1}
                  opacity={0.4}
                />
              )
            })}

            {/* Q heads */}
            {layout.qPositions.map((x, i) => (
              <g key={`q-${i}`}>
                <circle cx={x} cy={layout.qY} r={9} fill="#141420" stroke="#fbbf24" strokeWidth={1.6} />
                <text
                  x={x}
                  y={layout.qY + 3.5}
                  textAnchor="middle"
                  fontSize="7.5"
                  fill="#fbbf24"
                  fontFamily="JetBrains Mono, monospace"
                >
                  Q{i + 1}
                </text>
              </g>
            ))}

            {/* KV label */}
            <text x={20} y={255} fontSize="11" fill={regime.color} fontFamily="JetBrains Mono, monospace">
              kv heads · 1 → {n_kv}
            </text>

            {/* KV heads */}
            {layout.kvPositions.map((x, i) => (
              <g key={`kv-${i}`}>
                <rect
                  x={x - 16}
                  y={layout.kvY - 12}
                  width={32}
                  height={24}
                  rx={4}
                  fill="#1a1530"
                  stroke={regime.color}
                  strokeWidth={1.6}
                />
                <text
                  x={x}
                  y={layout.kvY + 3.5}
                  textAnchor="middle"
                  fontSize="9"
                  fill={regime.color}
                  fontFamily="JetBrains Mono, monospace"
                >
                  KV{i + 1}
                </text>
              </g>
            ))}

            {/* regime tag */}
            <g transform={`translate(${layout.W - 150}, 22)`}>
              <rect x={0} y={0} width={130} height={20} rx={4} fill="#141420" stroke={regime.color} strokeWidth={1} />
              <text x={65} y={14} textAnchor="middle" fontSize="11" fill={regime.color} fontFamily="JetBrains Mono, monospace">
                {regime.name}
              </text>
            </g>
          </svg>
        </div>

        <div className="rounded border border-dark-border bg-dark-surface-elevated/40 p-2.5 font-mono text-[10.5px] leading-relaxed text-dark-text-muted">
          <span style={{ color: regime.color }} className="font-semibold">{regime.name}:</span> {regime.desc}
        </div>
      </div>
    </WidgetFrame>
  )
}
