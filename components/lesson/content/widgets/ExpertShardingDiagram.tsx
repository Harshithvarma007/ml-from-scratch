'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// A visual map of expert parallelism. Experts are shared across GPUs: each
// GPU holds `E/G` experts. Tokens that live on GPU g may need to be routed
// to an expert that physically sits on a different GPU — that's the
// all-to-all. The diagram shows GPUs as boxes, experts inside them, and the
// inter-GPU traffic as arrows whose thickness reflects the volume.

const E_OPTIONS = [8, 16, 32]
const G_OPTIONS = [2, 4, 8]

// GPU colors cycle from this palette
const GPU_COLORS = ['#67e8f9', '#fbbf24', '#f472b6', '#4ade80', '#a78bfa', '#5eead4', '#fb7185', '#818cf8']

export default function ExpertShardingDiagram() {
  const [E, setE] = useState(8)
  const [G, setG] = useState(4)

  const expertsPerGpu = Math.ceil(E / G)

  // Stylized traffic model: each GPU's local tokens go mostly to other GPUs
  // (fraction ≈ (G-1)/G) and a small piece stays local. We use that to size
  // the all-to-all arrow widths.
  const traffic = useMemo(() => {
    const rows: number[][] = []
    for (let src = 0; src < G; src++) {
      const row: number[] = []
      for (let dst = 0; dst < G; dst++) {
        if (src === dst) {
          row.push(1 / G) // local share
        } else {
          // With uniform routing, each non-local GPU gets (1-1/G) / (G-1) of the tokens.
          row.push((1 - 1 / G) / Math.max(1, G - 1))
        }
      }
      rows.push(row)
    }
    return rows
  }, [G])

  const commVolFraction = (G - 1) / G

  // SVG layout
  const W = 900
  const H = 400
  const centerX = W / 2
  const centerY = H / 2 + 10
  const gpuR = 145 // ring radius

  const gpuPos = Array.from({ length: G }, (_, i) => {
    const theta = -Math.PI / 2 + (2 * Math.PI * i) / G
    return { x: centerX + gpuR * Math.cos(theta), y: centerY + gpuR * Math.sin(theta) }
  })

  const gpuBoxW = G === 2 ? 200 : G === 4 ? 160 : 120
  const gpuBoxH = 26 + 20 * expertsPerGpu

  return (
    <WidgetFrame
      widgetName="ExpertShardingDiagram"
      label="expert parallelism — experts live on different GPUs, tokens travel"
      right={
        <span className="font-mono">
          {E} experts · {G} GPUs · {expertsPerGpu} experts/GPU · all-to-all
        </span>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="num experts"
            value={E}
            min={8}
            max={32}
            step={8}
            onChange={(v) => setE(E_OPTIONS.reduce((b, c) => (Math.abs(c - v) < Math.abs(b - v) ? c : b)))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-cyan"
          />
          <Slider
            label="num GPUs"
            value={G}
            min={2}
            max={8}
            step={2}
            onChange={(v) => setG(G_OPTIONS.reduce((b, c) => (Math.abs(c - v) < Math.abs(b - v) ? c : b)))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="experts/GPU" value={String(expertsPerGpu)} accent="text-term-cyan" />
            <Readout
              label="inter-GPU traffic"
              value={`${(commVolFraction * 100).toFixed(0)}%`}
              accent="text-term-rose"
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full">
          <defs>
            <marker id="ep-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="#fb7185" />
            </marker>
          </defs>

          {/* Traffic arrows between GPUs */}
          {Array.from({ length: G }).flatMap((_, src) =>
            Array.from({ length: G }).map((__, dst) => {
              if (src === dst) return null
              const from = gpuPos[src]
              const to = gpuPos[dst]
              // Offset slightly to avoid overlap with node centers
              const dx = to.x - from.x
              const dy = to.y - from.y
              const len = Math.hypot(dx, dy)
              const ux = dx / len
              const uy = dy / len
              const pad = gpuBoxW / 2 + 6
              const x1 = from.x + ux * pad
              const y1 = from.y + uy * pad
              const x2 = to.x - ux * pad
              const y2 = to.y - uy * pad

              const v = traffic[src][dst]
              const sw = Math.max(0.8, v * 18)
              return (
                <line
                  key={`t-${src}-${dst}`}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke="#fb7185"
                  strokeWidth={sw}
                  opacity={0.55}
                  markerEnd="url(#ep-arrow)"
                />
              )
            }),
          )}

          {/* GPU nodes */}
          {gpuPos.map((p, g) => {
            const color = GPU_COLORS[g % GPU_COLORS.length]
            const bx = p.x - gpuBoxW / 2
            const by = p.y - gpuBoxH / 2
            // Which expert indices live on this GPU?
            const startE = g * expertsPerGpu
            const ids: number[] = []
            for (let e = startE; e < Math.min(E, startE + expertsPerGpu); e++) ids.push(e)
            return (
              <g key={`gpu-${g}`}>
                <rect
                  x={bx}
                  y={by}
                  width={gpuBoxW}
                  height={gpuBoxH}
                  rx={8}
                  fill="#0f0f1a"
                  stroke={color}
                  strokeWidth={1.8}
                />
                <text
                  x={p.x}
                  y={by + 16}
                  textAnchor="middle"
                  fontSize="11"
                  fill={color}
                  fontFamily="JetBrains Mono, monospace"
                  fontWeight={700}
                >
                  GPU {g}
                </text>
                {/* Experts inside */}
                {ids.map((e, idx) => (
                  <g key={`e-${e}`}>
                    <rect
                      x={bx + 10}
                      y={by + 22 + idx * 20}
                      width={gpuBoxW - 20}
                      height={16}
                      rx={3}
                      fill="#1a1a28"
                      stroke="#2a2a32"
                      strokeWidth={0.7}
                    />
                    <text
                      x={bx + 18}
                      y={by + 34 + idx * 20}
                      fontSize="9.5"
                      fill={color}
                      fontFamily="JetBrains Mono, monospace"
                    >
                      E{e}
                    </text>
                    <text
                      x={bx + gpuBoxW - 18}
                      y={by + 34 + idx * 20}
                      textAnchor="end"
                      fontSize="8.5"
                      fill="#6b7280"
                      fontFamily="JetBrains Mono, monospace"
                    >
                      MLP
                    </text>
                  </g>
                ))}
              </g>
            )
          })}

          {/* Center label */}
          <g>
            <circle cx={centerX} cy={centerY} r={44} fill="#0a0a12" stroke="#2a2a32" strokeWidth={1} />
            <text
              x={centerX}
              y={centerY - 4}
              textAnchor="middle"
              fontSize="10.5"
              fill="#fb7185"
              fontFamily="JetBrains Mono, monospace"
              fontWeight={700}
            >
              all-to-all
            </text>
            <text
              x={centerX}
              y={centerY + 10}
              textAnchor="middle"
              fontSize="9"
              fill="#a1a1aa"
              fontFamily="JetBrains Mono, monospace"
            >
              dispatch
            </text>
            <text
              x={centerX}
              y={centerY + 22}
              textAnchor="middle"
              fontSize="9"
              fill="#a1a1aa"
              fontFamily="JetBrains Mono, monospace"
            >
              + combine
            </text>
          </g>

          {/* Legend */}
          <g transform="translate(16, 14)">
            <line x1={0} y1={6} x2={14} y2={6} stroke="#fb7185" strokeWidth={4} opacity={0.6} />
            <text x={20} y={9} fontSize="9.5" fill="#ccc" fontFamily="JetBrains Mono, monospace">
              thicker = more tokens
            </text>
            <text x={0} y={24} fontSize="9" fill="#6b7280" fontFamily="JetBrains Mono, monospace">
              each GPU sends (G-1)/G of its tokens elsewhere
            </text>
          </g>

          <text
            x={W - 10}
            y={H - 10}
            textAnchor="end"
            fontSize="9"
            fill="#6b7280"
            fontFamily="JetBrains Mono, monospace"
          >
            comm per token ≈ 2 · d_model words (dispatch + combine)
          </text>
        </svg>
      </div>
    </WidgetFrame>
  )
}
