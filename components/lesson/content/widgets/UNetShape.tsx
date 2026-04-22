'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// SVG diagram of a U-Net at adjustable depth (3, 4, or 5 levels). Channels
// double each down-step starting from 64. The input resolution and channel
// count come from a dropdown. Clicking any block prints its exact
// (H × W × C) shape in the sidebar. Skip connections are rendered as
// dashed arcs between matching levels.

type InputCfg = { label: string; h: number; w: number; c: number }

const INPUTS: InputCfg[] = [
  { label: '32×32×3', h: 32, w: 32, c: 3 },
  { label: '64×64×3', h: 64, w: 64, c: 3 },
]

type Block = {
  id: string
  level: number
  side: 'down' | 'up' | 'bottom'
  h: number
  w: number
  c: number
  cx: number
  cy: number
}

function buildBlocks(depth: number, input: InputCfg): Block[] {
  const blocks: Block[] = []
  const baseC = 64
  const levels = depth // number of down blocks == number of up blocks
  // Center-vertical layout.
  const centerY = 200
  const levelDY = 46
  const leftX0 = 120
  const colGap = 70

  // Contracting path
  for (let l = 0; l < levels; l++) {
    const c = baseC * 2 ** l
    const h = input.h >> l
    const w = input.w >> l
    blocks.push({
      id: `down-${l}`,
      level: l,
      side: 'down',
      h,
      w,
      c,
      cx: leftX0 + l * colGap,
      cy: centerY + l * levelDY,
    })
  }
  // Bottleneck
  const bC = baseC * 2 ** levels
  const bH = input.h >> levels
  const bW = input.w >> levels
  blocks.push({
    id: 'bottom',
    level: levels,
    side: 'bottom',
    h: bH,
    w: bW,
    c: bC,
    cx: leftX0 + levels * colGap,
    cy: centerY + levels * levelDY,
  })
  // Expanding path
  for (let l = 0; l < levels; l++) {
    const lvl = levels - 1 - l
    const c = baseC * 2 ** lvl
    const h = input.h >> lvl
    const w = input.w >> lvl
    blocks.push({
      id: `up-${lvl}`,
      level: lvl,
      side: 'up',
      h,
      w,
      c,
      cx: leftX0 + (levels + 1 + l) * colGap,
      cy: centerY + lvl * levelDY,
    })
  }
  return blocks
}

export default function UNetShape() {
  const [depth, setDepth] = useState(4)
  const [inputIdx, setInputIdx] = useState(1)
  const [selected, setSelected] = useState<string>('bottom')

  const input = INPUTS[inputIdx]
  const blocks = useMemo(() => buildBlocks(depth, input), [depth, input])
  const sel = blocks.find((b) => b.id === selected) ?? blocks[0]

  const totalParams = blocks
    .filter((b) => b.side !== 'up')
    .reduce((acc, b) => acc + b.c * b.c * 9, 0) * 2 // rough 3x3 conv estimate

  return (
    <WidgetFrame
      widgetName="UNetShape"
      label="U-Net topology — click a block to inspect its tensor shape"
      right={<span className="font-mono">double channels on down, halve on up · skip connections reuse features</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="depth"
            value={depth}
            min={3}
            max={5}
            step={1}
            onChange={(v) => setDepth(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-cyan"
          />
          <div className="flex items-center gap-1.5">
            {INPUTS.map((cfg, i) => (
              <button
                key={cfg.label}
                onClick={() => setInputIdx(i)}
                className={cn(
                  'px-2 py-1 rounded text-[10.5px] font-mono transition-all',
                  inputIdx === i
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {cfg.label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="levels" value={String(depth)} accent="text-term-cyan" />
            <Readout
              label="bottleneck C"
              value={String(64 * 2 ** depth)}
              accent="text-term-amber"
            />
            <Readout
              label="~params"
              value={`${(totalParams / 1e6).toFixed(1)}M`}
              accent="text-term-green"
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-1 md:grid-cols-[1fr_220px] gap-4">
        <div className="relative min-w-0 min-h-0">
          <svg viewBox="0 0 800 460" className="w-full h-full">
            {/* Skip connection arcs */}
            {Array.from({ length: depth }).map((_, l) => {
              const down = blocks.find((b) => b.id === `down-${l}`)
              const up = blocks.find((b) => b.id === `up-${l}`)
              if (!down || !up) return null
              const cy = down.cy - 8
              return (
                <path
                  key={`skip-${l}`}
                  d={`M ${down.cx} ${cy} Q ${(down.cx + up.cx) / 2} ${cy - 60} ${up.cx} ${cy}`}
                  fill="none"
                  stroke="rgba(94, 234, 212, 0.45)"
                  strokeWidth={1.4}
                  strokeDasharray="4 3"
                />
              )
            })}

            {/* Connecting arrows down path */}
            {blocks.map((b, i) => {
              const next = blocks[i + 1]
              if (!next) return null
              return (
                <line
                  key={`conn-${i}`}
                  x1={b.cx}
                  y1={b.cy}
                  x2={next.cx}
                  y2={next.cy}
                  stroke="#2a2a32"
                  strokeWidth={1.2}
                />
              )
            })}

            {/* Blocks */}
            {blocks.map((b) => {
              const isSel = b.id === selected
              const color =
                b.side === 'down' ? '#67e8f9' : b.side === 'up' ? '#f472b6' : '#fbbf24'
              const size = 26 + b.c / 32 // scale with channels
              return (
                <g
                  key={b.id}
                  onClick={() => setSelected(b.id)}
                  style={{ cursor: 'pointer' }}
                >
                  <rect
                    x={b.cx - size / 2}
                    y={b.cy - 12}
                    width={size}
                    height={22}
                    rx={4}
                    fill={isSel ? '#1f1f2c' : '#141420'}
                    stroke={color}
                    strokeWidth={isSel ? 2 : 1.2}
                  />
                  <text
                    x={b.cx}
                    y={b.cy + 2}
                    textAnchor="middle"
                    fontSize="9"
                    fill={color}
                    fontFamily="JetBrains Mono, monospace"
                  >
                    C={b.c}
                  </text>
                </g>
              )
            })}

            {/* Legend */}
            <g transform="translate(20, 420)">
              <rect x={0} y={-8} width={12} height={12} fill="#141420" stroke="#67e8f9" />
              <text x={16} y={2} fontSize="10" fill="#67e8f9" fontFamily="JetBrains Mono, monospace">
                contracting
              </text>
              <rect x={110} y={-8} width={12} height={12} fill="#141420" stroke="#fbbf24" />
              <text x={126} y={2} fontSize="10" fill="#fbbf24" fontFamily="JetBrains Mono, monospace">
                bottleneck
              </text>
              <rect x={210} y={-8} width={12} height={12} fill="#141420" stroke="#f472b6" />
              <text x={226} y={2} fontSize="10" fill="#f472b6" fontFamily="JetBrains Mono, monospace">
                expanding
              </text>
              <line x1={320} y1={-2} x2={350} y2={-2} stroke="#5eead4" strokeWidth={1.4} strokeDasharray="4 3" />
              <text x={356} y={2} fontSize="10" fill="#5eead4" fontFamily="JetBrains Mono, monospace">
                skip
              </text>
            </g>
          </svg>
        </div>

        {/* Inspector */}
        <div className="flex flex-col gap-3 min-w-0 min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            inspector
          </div>
          <div className="bg-dark-surface-elevated/40 rounded border border-dark-border p-3 font-mono text-[11px] leading-relaxed">
            <div className="text-term-amber uppercase tracking-wider text-[9.5px]">block</div>
            <div className="text-dark-text-primary mb-2">{sel.id}</div>
            <div className="text-term-cyan uppercase tracking-wider text-[9.5px]">tensor shape</div>
            <div className="text-dark-text-primary text-[13px] tabular-nums mb-2">
              {sel.h} × {sel.w} × {sel.c}
            </div>
            <div className="text-term-green uppercase tracking-wider text-[9.5px]">spatial</div>
            <div className="text-dark-text-secondary mb-2">
              {sel.h * sel.w} pixels · stride 2^{sel.level}
            </div>
            <div className="text-term-purple uppercase tracking-wider text-[9.5px]">features</div>
            <div className="text-dark-text-secondary">{sel.c} channels</div>
          </div>

          <div className="text-[10px] font-mono text-dark-text-muted leading-snug mt-auto">
            <span className="text-term-cyan">skip</span> pipes the encoder feature map
            straight into the matching decoder stage — lets fine detail survive the
            bottleneck.
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
