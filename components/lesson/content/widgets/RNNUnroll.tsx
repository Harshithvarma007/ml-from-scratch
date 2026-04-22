'use client'

import { useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// An unrolled RNN drawn as 5 identical cells in a row. Click a cell to make
// it the active step — the picture highlights which h_{t-1} feeds in, which
// x_t feeds in, and which h_t comes out. The W_x / W_h / b block is drawn
// inside every cell to hammer home "same weights, every step."

const STEPS = 5
const TOKENS = ['h', 'e', 'l', 'l', 'o']

// A tiny deterministic RNN so the readouts look alive. h_t = tanh(wx * x + wh * h + b)
const W_X = 0.85
const W_H = 0.7
const B = -0.1

function hiddenTrace(): number[] {
  const out: number[] = [0]
  let h = 0
  // Map each char to a scalar between -1 and 1 deterministically
  TOKENS.forEach((t) => {
    const x = ((t.charCodeAt(0) - 100) / 10) // spans roughly -0.8 .. +1.1
    h = Math.tanh(W_X * x + W_H * h + B)
    out.push(h)
  })
  return out
}

const TRACE = hiddenTrace()

export default function RNNUnroll() {
  const [active, setActive] = useState(2)

  const hPrev = TRACE[active]
  const hCur = TRACE[active + 1]
  const xCur = ((TOKENS[active].charCodeAt(0) - 100) / 10)

  return (
    <WidgetFrame
      widgetName="RNNUnroll"
      label="unrolled RNN — one cell, called T times"
      right={<span className="font-mono">h_t = tanh(W_x·x_t + W_h·h_(t-1) + b)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="text-[11px] font-mono text-dark-text-muted">
            click a time step to highlight the recurrence
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="t" value={String(active + 1)} accent="text-term-amber" />
            <Readout label="x_t" value={xCur.toFixed(2)} />
            <Readout label="h_(t-1)" value={hPrev.toFixed(3)} />
            <Readout label="h_t" value={hCur.toFixed(3)} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 flex items-center justify-center overflow-auto">
        <svg viewBox="0 0 980 300" className="w-full max-w-[1100px] h-full">
          {/* Initial hidden state block on the far left */}
          <g>
            <circle cx={40} cy={160} r={18} fill="#1f1b3a" stroke="#67e8f9" strokeWidth={1.4} />
            <text x={40} y={165} textAnchor="middle" fontSize="11" fill="#67e8f9" fontFamily="JetBrains Mono, monospace">
              h₀
            </text>
          </g>

          {Array.from({ length: STEPS }).map((_, t) => {
            const x0 = 80 + t * 180
            const cellX = x0 + 30
            const cellY = 120
            const cellW = 120
            const cellH = 90
            const midX = cellX + cellW / 2
            const midY = cellY + cellH / 2
            const isActive = t === active
            const stroke = isActive ? '#fbbf24' : '#3f3f46'
            const textColor = isActive ? '#fbbf24' : '#a1a1aa'
            const dim = isActive ? 1 : 0.35
            return (
              <g
                key={t}
                onClick={() => setActive(t)}
                style={{ cursor: 'pointer' }}
              >
                {/* h from previous step → into cell */}
                <line
                  x1={x0}
                  y1={160}
                  x2={cellX}
                  y2={midY}
                  stroke={isActive ? '#4ade80' : '#3a3a44'}
                  strokeWidth={isActive ? 2.4 : 1.4}
                  markerEnd={isActive ? 'url(#arrow-green)' : 'url(#arrow-dim)'}
                />

                {/* x_t feeding up */}
                <g opacity={dim}>
                  <rect
                    x={midX - 18}
                    y={cellY + cellH + 24}
                    width={36}
                    height={26}
                    rx={4}
                    fill="#111827"
                    stroke={isActive ? '#67e8f9' : '#3f3f46'}
                    strokeWidth={1.2}
                  />
                  <text
                    x={midX}
                    y={cellY + cellH + 42}
                    textAnchor="middle"
                    fontSize="12"
                    fill={isActive ? '#67e8f9' : '#a1a1aa'}
                    fontFamily="JetBrains Mono, monospace"
                  >
                    {TOKENS[t]}
                  </text>
                </g>
                <line
                  x1={midX}
                  y1={cellY + cellH + 24}
                  x2={midX}
                  y2={cellY + cellH}
                  stroke={isActive ? '#67e8f9' : '#3a3a44'}
                  strokeWidth={isActive ? 2.2 : 1.2}
                  markerEnd={isActive ? 'url(#arrow-cyan)' : 'url(#arrow-dim)'}
                />
                <text
                  x={midX + 22}
                  y={cellY + cellH + 42}
                  fontSize="10"
                  fill="#555"
                  fontFamily="JetBrains Mono, monospace"
                >
                  x_{t + 1}
                </text>

                {/* Cell body */}
                <rect
                  x={cellX}
                  y={cellY}
                  width={cellW}
                  height={cellH}
                  rx={8}
                  fill={isActive ? '#1f2937' : '#141420'}
                  stroke={stroke}
                  strokeWidth={isActive ? 2 : 1.2}
                />
                <text
                  x={midX}
                  y={cellY + 20}
                  textAnchor="middle"
                  fontSize="10"
                  fill={textColor}
                  fontFamily="JetBrains Mono, monospace"
                  fontWeight={600}
                >
                  t = {t + 1}
                </text>
                <text
                  x={midX}
                  y={cellY + 42}
                  textAnchor="middle"
                  fontSize="11"
                  fill={isActive ? '#ffffff' : '#6b7280'}
                  fontFamily="JetBrains Mono, monospace"
                >
                  tanh
                </text>
                {/* Shared-params glyph */}
                <rect
                  x={midX - 30}
                  y={cellY + 54}
                  width={60}
                  height={20}
                  rx={3}
                  fill="#0f0f1a"
                  stroke={isActive ? '#a78bfa' : '#3f3f46'}
                  strokeWidth={1}
                />
                <text
                  x={midX}
                  y={cellY + 68}
                  textAnchor="middle"
                  fontSize="9.5"
                  fill={isActive ? '#a78bfa' : '#6b7280'}
                  fontFamily="JetBrains Mono, monospace"
                >
                  W_x W_h b
                </text>

                {/* h_t emerging from the top of the cell */}
                <text
                  x={midX}
                  y={cellY - 12}
                  textAnchor="middle"
                  fontSize="11"
                  fill={isActive ? '#4ade80' : '#555'}
                  fontFamily="JetBrains Mono, monospace"
                >
                  h_{t + 1} = {TRACE[t + 1].toFixed(2)}
                </text>
              </g>
            )
          })}

          {/* Terminal arrow past the last cell */}
          <line
            x1={80 + STEPS * 180 - 30}
            y1={160}
            x2={80 + STEPS * 180 + 10}
            y2={160}
            stroke="#3a3a44"
            strokeWidth={1.4}
          />
          <text
            x={80 + STEPS * 180 + 20}
            y={164}
            fontSize="11"
            fill="#777"
            fontFamily="JetBrains Mono, monospace"
          >
            h_T → output
          </text>

          <defs>
            <marker
              id="arrow-green"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M0,0 L10,5 L0,10 z" fill="#4ade80" />
            </marker>
            <marker
              id="arrow-cyan"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M0,0 L10,5 L0,10 z" fill="#67e8f9" />
            </marker>
            <marker
              id="arrow-dim"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path d="M0,0 L10,5 L0,10 z" fill="#3a3a44" />
            </marker>
          </defs>

          {/* Footer formula anchored to active step */}
          <g transform={`translate(${80 + active * 180 - 40}, 262)`}>
            <rect
              x={0}
              y={0}
              width={360}
              height={26}
              rx={4}
              fill="#0f0f1a"
              stroke="#fbbf24"
              strokeWidth={1}
              opacity={0.85}
            />
            <text
              x={180}
              y={17}
              textAnchor="middle"
              fontSize="11"
              fill="#fbbf24"
              fontFamily="JetBrains Mono, monospace"
            >
              h_{active + 1} = tanh({W_X}·{xCur.toFixed(2)} + {W_H}·{hPrev.toFixed(2)} + {B}) ={' '}
              {hCur.toFixed(3)}
            </text>
          </g>
        </svg>
      </div>
    </WidgetFrame>
  )
}
