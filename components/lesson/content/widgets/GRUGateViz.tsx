'use client'

import { useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Assemble a GRU update by hand. Two gates: r_t (reset) and z_t (update).
// Inputs: x_t and h_{t-1}. We model the candidate as tanh(w_x·x + w_h·r·h + b)
// and the final state as h_t = (1−z)·h_{t-1} + z·ĥ_t. The widget draws a
// horizontal interpolation ruler between h_{t-1} and ĥ_t with z marking
// where h_t lands — making the "linear interpolation" description literal.

export default function GRUGateViz() {
  const [r, setR] = useState(0.6)
  const [z, setZ] = useState(0.4)
  const [hPrev, setHPrev] = useState(0.7)
  const [x, setX] = useState(0.5)

  // Fixed "learned" params so the candidate looks sensible
  const wx = 0.7
  const wh = 0.8
  const b = -0.1

  const resetState = r * hPrev
  const candidate = Math.tanh(wx * x + wh * resetState + b)
  const h = (1 - z) * hPrev + z * candidate

  // Highlighted presets
  const presets = [
    { name: 'pure keep',      r: 1.0, z: 0.0, hPrev: 0.8, x: 0.3, why: 'z=0 — copy h_(t-1) forward, ignore candidate.' },
    { name: 'pure write',     r: 0.0, z: 1.0, hPrev: 0.8, x: 0.3, why: 'z=1 — overwrite memory with a candidate that ignores history (r=0).' },
    { name: 'blended',        r: 0.7, z: 0.5, hPrev: 0.5, x: 0.4, why: 'both gates open halfway — standard operating mode.' },
    { name: 'topic switch',   r: 0.1, z: 0.9, hPrev: 0.8, x: -0.6, why: 'r low (ignore prior context) + z high (commit to new state).' },
  ] as const

  return (
    <WidgetFrame
      widgetName="GRUGateViz"
      label="GRU — two gates, one interpolation"
      right={<span className="font-mono">h_t = (1−z)·h_(t-1) + z·ĥ_t · ĥ_t = tanh(W_x·x + W_h·(r⊙h_(t-1)) + b)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-3">
          {presets.map((p) => (
            <button
              key={p.name}
              onClick={() => { setR(p.r); setZ(p.z); setHPrev(p.hPrev); setX(p.x) }}
              className="px-2 py-1 rounded text-[10.5px] font-mono uppercase border border-dark-border text-dark-text-secondary hover:text-dark-text-primary transition-all"
            >
              {p.name}
            </button>
          ))}
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="ĥ_t" value={candidate.toFixed(3)} accent="text-term-purple" />
            <Readout label="h_t" value={h.toFixed(3)} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 grid grid-cols-1 md:grid-cols-[260px_1fr] gap-6 overflow-auto">
        {/* Sliders */}
        <div className="flex flex-col gap-3 min-w-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">inputs</div>
          <Slider label="x_t" value={x} min={-1} max={1} step={0.01} onChange={setX} format={(v) => v.toFixed(2)} accent="accent-term-cyan" />
          <Slider label="h_(t-1)" value={hPrev} min={-1} max={1} step={0.01} onChange={setHPrev} format={(v) => v.toFixed(2)} accent="accent-term-amber" />
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-1">gates</div>
          <Slider label="r_t  reset" value={r} min={0} max={1} step={0.01} onChange={setR} format={(v) => v.toFixed(2)} accent="accent-term-rose" />
          <Slider label="z_t  update" value={z} min={0} max={1} step={0.01} onChange={setZ} format={(v) => v.toFixed(2)} accent="accent-term-green" />

          <div className="text-[11px] font-mono text-dark-text-muted leading-relaxed mt-2">
            <span className="text-term-rose">r</span> gates what the candidate sees of history.{' '}
            <span className="text-term-green">z</span> mixes old state with new candidate.
          </div>
        </div>

        {/* Flow diagram + interpolation ruler */}
        <div className="flex flex-col gap-4 min-w-0">
          <svg viewBox="0 0 620 240" className="w-full">
            {/* Previous hidden */}
            <circle cx={50} cy={120} r={24} fill="#1f1b3a" stroke="#fbbf24" strokeWidth={1.4} />
            <text x={50} y={124} textAnchor="middle" fontSize="10" fill="#fbbf24" fontFamily="JetBrains Mono, monospace">
              h_(t-1)
            </text>
            <text x={50} y={158} textAnchor="middle" fontSize="10" fill="#fbbf24" fontFamily="JetBrains Mono, monospace">
              {hPrev.toFixed(2)}
            </text>

            {/* x_t input */}
            <rect x={22} y={18} width={56} height={26} rx={4} fill="#111827" stroke="#67e8f9" strokeWidth={1.2} />
            <text x={50} y={36} textAnchor="middle" fontSize="10.5" fill="#67e8f9" fontFamily="JetBrains Mono, monospace">
              x_t = {x.toFixed(2)}
            </text>

            {/* r gate */}
            <g>
              <line x1={80} y1={120} x2={140} y2={120} stroke="#f87171" strokeWidth={1.6} />
              <circle cx={170} cy={120} r={20} fill="#1a1a1a" stroke="#f87171" strokeWidth={1.4} />
              <text x={170} y={118} textAnchor="middle" fontSize="10" fill="#f87171" fontFamily="JetBrains Mono, monospace">
                r⊙
              </text>
              <text x={170} y={131} textAnchor="middle" fontSize="9" fill="#f87171" fontFamily="JetBrains Mono, monospace">
                {r.toFixed(2)}
              </text>
            </g>

            {/* Candidate cell */}
            <g>
              <line x1={190} y1={120} x2={230} y2={120} stroke="#a78bfa" strokeWidth={1.6} />
              <line x1={50} y1={46} x2={50} y2={96} stroke="#67e8f9" strokeWidth={1.2} />
              <line x1={50} y1={96} x2={230} y2={96} stroke="#67e8f9" strokeWidth={1.2} />
              <line x1={230} y1={96} x2={230} y2={110} stroke="#67e8f9" strokeWidth={1.2} />
              <rect x={230} y={96} width={130} height={48} rx={6} fill="#141420" stroke="#a78bfa" strokeWidth={1.4} />
              <text x={295} y={116} textAnchor="middle" fontSize="10" fill="#a78bfa" fontFamily="JetBrains Mono, monospace">
                ĥ_t = tanh(·)
              </text>
              <text x={295} y={134} textAnchor="middle" fontSize="11" fill="#e5e7eb" fontFamily="JetBrains Mono, monospace">
                {candidate.toFixed(3)}
              </text>
            </g>

            {/* z gate interpolation node */}
            <g>
              <line x1={360} y1={120} x2={420} y2={120} stroke="#a78bfa" strokeWidth={1.6} />
              <line x1={80} y1={120} x2={80} y2={196} stroke="#fbbf24" strokeWidth={1.2} />
              <line x1={80} y1={196} x2={440} y2={196} stroke="#fbbf24" strokeWidth={1.2} />
              <line x1={440} y1={196} x2={440} y2={140} stroke="#fbbf24" strokeWidth={1.2} />
              <rect x={420} y={100} width={60} height={44} rx={6} fill="#141420" stroke="#4ade80" strokeWidth={1.4} />
              <text x={450} y={118} textAnchor="middle" fontSize="10" fill="#4ade80" fontFamily="JetBrains Mono, monospace">
                mix
              </text>
              <text x={450} y={134} textAnchor="middle" fontSize="9" fill="#4ade80" fontFamily="JetBrains Mono, monospace">
                z = {z.toFixed(2)}
              </text>
            </g>

            {/* Output */}
            <line x1={480} y1={120} x2={540} y2={120} stroke="#4ade80" strokeWidth={1.8} />
            <circle cx={568} cy={120} r={26} fill="#0f1f14" stroke="#4ade80" strokeWidth={1.6} />
            <text x={568} y={117} textAnchor="middle" fontSize="10" fill="#4ade80" fontFamily="JetBrains Mono, monospace">
              h_t
            </text>
            <text x={568} y={132} textAnchor="middle" fontSize="11" fill="#ffffff" fontFamily="JetBrains Mono, monospace">
              {h.toFixed(3)}
            </text>
          </svg>

          {/* Interpolation ruler */}
          <div className="mt-1">
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
              h_t lives on this line — z_t is where it lands
            </div>
            <InterpRuler hPrev={hPrev} candidate={candidate} z={z} h={h} />
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function InterpRuler({
  hPrev,
  candidate,
  z,
  h,
}: {
  hPrev: number
  candidate: number
  z: number
  h: number
}) {
  // Show a [-1, 1] ruler with hPrev (amber), candidate (purple), and h (green)
  // drawn at their x-positions. A ribbon between hPrev and candidate shows the
  // reachable interpolation range for any z ∈ [0, 1].
  const toPct = (v: number) => 50 + v * 50
  const minX = Math.min(hPrev, candidate)
  const maxX = Math.max(hPrev, candidate)

  return (
    <div className="relative h-16 bg-dark-surface-elevated/40 rounded overflow-visible">
      {/* Axis */}
      <div className="absolute inset-x-3 top-1/2 h-px bg-dark-border" />
      {[-1, -0.5, 0, 0.5, 1].map((v) => (
        <div
          key={v}
          className="absolute top-1/2 -translate-y-1/2 text-[9px] font-mono text-dark-text-disabled"
          style={{ left: `calc(${toPct(v)}% - 10px)`, transform: 'translateY(18px)' }}
        >
          {v.toFixed(1)}
        </div>
      ))}

      {/* Reachable interpolation ribbon */}
      <div
        className="absolute top-1/2 -translate-y-1/2 h-1 bg-term-purple/35 rounded-full"
        style={{
          left: `${toPct(minX)}%`,
          width: `${toPct(maxX) - toPct(minX)}%`,
        }}
      />

      {/* h_prev marker */}
      <Marker x={toPct(hPrev)} label="h_(t-1)" color="#fbbf24" />
      {/* candidate marker */}
      <Marker x={toPct(candidate)} label="ĥ_t" color="#a78bfa" offsetY={-18} />
      {/* h marker */}
      <Marker x={toPct(h)} label="h_t" color="#4ade80" offsetY={18} solid />

      {/* z slider arrow */}
      <div
        className="absolute top-1/2 -translate-y-1/2 text-[9px] font-mono text-term-green"
        style={{ left: `calc(${toPct(h)}% - 18px)`, transform: 'translate(0, -30px)' }}
      >
        z = {z.toFixed(2)}
      </div>
    </div>
  )
}

function Marker({
  x,
  label,
  color,
  offsetY = 0,
  solid = false,
}: {
  x: number
  label: string
  color: string
  offsetY?: number
  solid?: boolean
}) {
  return (
    <div
      className={cn('absolute top-1/2 -translate-y-1/2 -translate-x-1/2 flex flex-col items-center gap-0.5')}
      style={{ left: `${x}%` }}
    >
      <div
        className="w-2.5 h-2.5 rounded-full border-2"
        style={{ borderColor: color, backgroundColor: solid ? color : '#141420' }}
      />
      <div
        className="text-[9.5px] font-mono whitespace-nowrap"
        style={{ color, transform: `translateY(${offsetY}px)` }}
      >
        {label}
      </div>
    </div>
  )
}
