'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Plot all-to-all communication cost for an MoE layer against the compute
// cost of the layer itself. As you scale d_model, the compute grows ~d² but
// all-to-all traffic grows ~d, so communication becomes a smaller *fraction*
// of compute. Scale G instead and the comm tax goes up. The plot draws both
// curves across d_model and marks the crossover where comm > compute.

// Per-token quantities, in FLOPs for compute and words (values of dtype size)
// for communication. We assume an MoE block with a single MLP expert of
// dims d_model → d_ff → d_model, d_ff = 4 · d_model. Two all-to-all rounds
// per block (dispatch + combine), each shipping one d_model vector per token.

const BANDWIDTH_GB_S = 600          // realistic NVLink-ish single-direction BW
const FLOPS_S = 400e12              // 400 TFLOP/s per GPU, fp16
const BYTES_PER_WORD = 2            // fp16

const D_OPTIONS = [1024, 2048, 4096, 8192, 16384]
const G_OPTIONS = [2, 4, 8, 16, 64]

// compute per token (two matmuls: d · 4d + 4d · d = 8d²)
function computeFlops(d: number): number {
  return 8 * d * d
}

// all-to-all bytes per token
function commBytes(d: number, G: number): number {
  // dispatch + combine, each (G-1)/G × d_model words
  return 2 * ((G - 1) / G) * d * BYTES_PER_WORD
}

function computeSec(d: number): number {
  return computeFlops(d) / FLOPS_S
}

function commSec(d: number, G: number): number {
  return commBytes(d, G) / (BANDWIDTH_GB_S * 1e9)
}

export default function AllToAllCost() {
  const [d, setD] = useState(4096)
  const [G, setG] = useState(8)

  const pts = useMemo(() => {
    const xs: number[] = []
    for (let dd = 512; dd <= 32768; dd *= 2) xs.push(dd)
    return xs.map((dd) => ({
      d: dd,
      compute: computeSec(dd) * 1e9, // ns per token
      comm: commSec(dd, G) * 1e9,    // ns per token
    }))
  }, [G])

  const currCompute = computeSec(d) * 1e9
  const currComm = commSec(d, G) * 1e9
  const commFrac = currComm / (currCompute + currComm)

  // Find crossover
  const crossover = pts.find((p, i) => i > 0 && pts[i - 1].comm > pts[i - 1].compute && p.comm <= p.compute)
    ?? pts.find((p, i) => i > 0 && pts[i - 1].comm < pts[i - 1].compute && p.comm >= p.compute)

  return (
    <WidgetFrame
      widgetName="AllToAllCost"
      label="all-to-all vs MLP compute — where does the comm tax win?"
      right={
        <span className="font-mono">
          fp16 · {BANDWIDTH_GB_S} GB/s · {FLOPS_S / 1e12} TFLOP/s
        </span>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="d_model"
            value={d}
            min={1024}
            max={16384}
            step={1024}
            onChange={(v) => setD(D_OPTIONS.reduce((b, c) => (Math.abs(c - v) < Math.abs(b - v) ? c : b)))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-cyan"
          />
          <Slider
            label="num GPUs"
            value={G}
            min={2}
            max={64}
            step={1}
            onChange={(v) => setG(G_OPTIONS.reduce((b, c) => (Math.abs(c - v) < Math.abs(b - v) ? c : b)))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="compute" value={`${currCompute.toFixed(1)} ns`} accent="text-term-green" />
            <Readout label="all-to-all" value={`${currComm.toFixed(1)} ns`} accent="text-term-rose" />
            <Readout
              label="comm %"
              value={`${(commFrac * 100).toFixed(0)}%`}
              accent={commFrac > 0.5 ? 'text-term-rose' : 'text-term-green'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-[1.3fr_1fr] gap-4">
        <Plot pts={pts} d={d} crossover={crossover?.d ?? null} />
        <RightPanel d={d} G={G} compute={currComm + currCompute} commFrac={commFrac} currCompute={currCompute} currComm={currComm} />
      </div>
    </WidgetFrame>
  )
}

function Plot({
  pts,
  d,
  crossover,
}: {
  pts: { d: number; compute: number; comm: number }[]
  d: number
  crossover: number | null
}) {
  const W = 520
  const H = 300
  const padL = 46
  const padR = 14
  const padT = 34
  const padB = 40
  const plotW = W - padL - padR
  const plotH = H - padT - padB

  const xMin = Math.log2(pts[0].d)
  const xMax = Math.log2(pts[pts.length - 1].d)
  const yMax = Math.max(...pts.map((p) => Math.max(p.compute, p.comm))) * 1.1
  const toSx = (dd: number) => padL + ((Math.log2(dd) - xMin) / (xMax - xMin)) * plotW
  const toSy = (v: number) => padT + plotH - (Math.log10(1 + v) / Math.log10(1 + yMax)) * plotH

  const pathComp = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${toSx(p.d).toFixed(1)} ${toSy(p.compute).toFixed(1)}`).join(' ')
  const pathComm = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${toSx(p.d).toFixed(1)} ${toSy(p.comm).toFixed(1)}`).join(' ')

  // Ticks
  const yTicks = [1, 10, 100, 1000, 10000]
  const xTicks = [512, 2048, 8192, 32768]

  return (
    <div className="flex flex-col gap-1 min-h-0">
      <div className="flex items-baseline justify-between font-mono">
        <div className="text-[11px] text-dark-text-primary">per-token latency (log axes)</div>
        <div className="text-[9.5px] text-dark-text-muted">compute grows d² · comm grows d</div>
      </div>
      <div className="flex-1 min-h-0">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="w-full h-full">
          {yTicks.filter((t) => t <= yMax).map((t) => (
            <g key={`yt-${t}`}>
              <line x1={padL} y1={toSy(t)} x2={W - padR} y2={toSy(t)} stroke="#1e1e1e" strokeWidth={1} />
              <text x={padL - 6} y={toSy(t) + 3} textAnchor="end" fontSize="9" fill="#555" fontFamily="JetBrains Mono, monospace">
                {t >= 1000 ? `${t / 1000}µs` : `${t}ns`}
              </text>
            </g>
          ))}
          {xTicks.map((x) => (
            <text
              key={`xt-${x}`}
              x={toSx(x)}
              y={H - 18}
              textAnchor="middle"
              fontSize="9"
              fill="#555"
              fontFamily="JetBrains Mono, monospace"
            >
              {x}
            </text>
          ))}
          <text
            x={padL + plotW / 2}
            y={H - 4}
            textAnchor="middle"
            fontSize="9"
            fill="#777"
            fontFamily="JetBrains Mono, monospace"
          >
            d_model
          </text>

          {/* Crossover highlight */}
          {crossover !== null && (
            <line
              x1={toSx(crossover)}
              y1={padT}
              x2={toSx(crossover)}
              y2={padT + plotH}
              stroke="#fbbf24"
              strokeDasharray="3 3"
              strokeWidth={1}
              opacity={0.7}
            />
          )}

          {/* Lines */}
          <path d={pathComp} stroke="#4ade80" strokeWidth={2} fill="none" />
          <path d={pathComm} stroke="#fb7185" strokeWidth={2} fill="none" />

          {/* Current cursor */}
          <line
            x1={toSx(d)}
            y1={padT}
            x2={toSx(d)}
            y2={padT + plotH}
            stroke="rgba(255,255,255,0.35)"
            strokeDasharray="2 3"
            strokeWidth={1}
          />
          <circle cx={toSx(d)} cy={toSy(computeSec(d) * 1e9)} r={4} fill="#4ade80" />
          <circle cx={toSx(d)} cy={toSy(commSec(d, 8) * 1e9)} r={4} fill="#fb7185" />

          {/* Legend */}
          <g transform={`translate(${padL + 6}, ${padT - 2})`}>
            <line x1={0} y1={6} x2={14} y2={6} stroke="#4ade80" strokeWidth={2} />
            <text x={18} y={9} fontSize="9.5" fill="#ccc" fontFamily="JetBrains Mono, monospace">
              MLP compute (8 · d² FLOPs)
            </text>
            <line x1={180} y1={6} x2={194} y2={6} stroke="#fb7185" strokeWidth={2} />
            <text x={198} y={9} fontSize="9.5" fill="#ccc" fontFamily="JetBrains Mono, monospace">
              all-to-all (2d words)
            </text>
          </g>
        </svg>
      </div>
    </div>
  )
}

function RightPanel({
  d,
  G,
  commFrac,
  currCompute,
  currComm,
}: {
  d: number
  G: number
  compute: number
  commFrac: number
  currCompute: number
  currComm: number
}) {
  const ratio = currComm / currCompute
  const dpStr = `${d.toLocaleString()} × ${(G - 1).toLocaleString()} / ${G}`
  return (
    <div className="flex flex-col gap-3 min-h-0 min-w-0">
      <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
        breakdown at the cursor
      </div>

      {/* Stacked bar */}
      <div className="bg-dark-surface-elevated/40 rounded p-3">
        <div className="font-mono text-[10px] text-dark-text-muted mb-1.5">per-token latency</div>
        <div className="h-5 rounded overflow-hidden flex bg-dark-bg">
          <div
            className="h-full"
            style={{ width: `${(1 - commFrac) * 100}%`, backgroundColor: '#4ade80', opacity: 0.85 }}
          />
          <div
            className="h-full"
            style={{ width: `${commFrac * 100}%`, backgroundColor: '#fb7185', opacity: 0.85 }}
          />
        </div>
        <div className="flex items-center justify-between mt-1 font-mono text-[9.5px]">
          <span className="text-term-green">compute {currCompute.toFixed(1)}ns</span>
          <span className="text-term-rose">all-to-all {currComm.toFixed(1)}ns</span>
        </div>
      </div>

      <div className="bg-dark-surface-elevated/40 rounded p-3 font-mono text-[10.5px] leading-relaxed space-y-1.5">
        <div className="text-dark-text-secondary">compute per token</div>
        <div className="text-term-green">≈ 8 · d² = {(8 * d * d).toExponential(2)} FLOPs</div>
        <div className="text-dark-text-secondary mt-1.5">comm per token</div>
        <div className="text-term-rose">
          ≈ 2 · (G-1)/G · d = 2 · {dpStr} words
        </div>
        <div className="text-dark-text-disabled mt-1.5">
          comm/compute = {ratio.toFixed(2)}×
        </div>
      </div>

      <div className="bg-dark-surface-elevated/40 rounded p-3 font-mono text-[10.5px] leading-relaxed">
        <div className="text-term-amber mb-1">takeaway</div>
        <div className="text-dark-text-muted">
          {commFrac > 0.5
            ? 'comm dominates: add bandwidth, reduce G, or shrink d_model.'
            : commFrac > 0.3
              ? 'comm is a meaningful tax — scale d and the ratio improves.'
              : 'compute-bound: going d²→ is why bigger models hide all-to-all well.'}
        </div>
      </div>
    </div>
  )
}
