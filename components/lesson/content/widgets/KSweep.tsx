'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Two coupled plots across k = 1 .. 8. Left: FLOPs per token rise linearly;
// quality rises sharply k=1 → 2, then flattens. Right: routing entropy
// (− Σ p log p) keeps rising — with more slots to fill, the router spreads
// probability wider. The useful region is the sliver around k = 2.

const NUM_EXPERTS = 8

// Per-token FLOPs proportional to k — use "1.0" as base unit at k=1.
function flopsAt(k: number): number {
  return k
}

// A stylized "quality" curve. Calibrated so k=1 ≈ 0.72, k=2 ≈ 0.92, k=4 ≈
// 0.95, k=8 ≈ 0.97. Matches Switch/Shazeer style findings.
function qualityAt(k: number): number {
  return 0.72 + 0.25 * (1 - Math.exp(-0.9 * (k - 1)))
}

// Entropy of a top-k softmax: the router, when it has to populate k slots,
// tends to spread probability. Stylized: H = log2(k) + small curvature.
function entropyAt(k: number): number {
  return k === 1 ? 0 : Math.log2(k) + 0.15 * (1 - Math.exp(-(k - 1) / 2))
}

export default function KSweep() {
  const [k, setK] = useState(2)
  const flops = flopsAt(k)
  const quality = qualityAt(k)
  const entropy = entropyAt(k)

  const qualityGain = (qualityAt(2) - qualityAt(1)) * 100
  const flopsCost = (flopsAt(2) - flopsAt(1)) * 100

  const series = useMemo(() => {
    const xs: number[] = []
    for (let kk = 1; kk <= NUM_EXPERTS; kk++) xs.push(kk)
    return {
      xs,
      flops: xs.map(flopsAt),
      quality: xs.map(qualityAt),
      entropy: xs.map(entropyAt),
    }
  }, [])

  return (
    <WidgetFrame
      widgetName="KSweep"
      label="k sweep — quality saturates, FLOPs keep climbing"
      right={<span className="font-mono">E = {NUM_EXPERTS} experts · k=2 is the sweet spot</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="k"
            value={k}
            min={1}
            max={NUM_EXPERTS}
            step={1}
            onChange={(v) => setK(Math.round(v))}
            format={(v) => `top-${Math.round(v)}`}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="FLOPs" value={`${flops.toFixed(1)}×`} accent="text-term-rose" />
            <Readout label="quality" value={quality.toFixed(3)} accent="text-term-green" />
            <Readout label="H(router)" value={`${entropy.toFixed(2)} bits`} accent="text-term-cyan" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-2 gap-4">
        {/* Plot 1: FLOPs + quality share a y-axis (quality rescaled for overlay) */}
        <Plot
          title="FLOPs per token  vs  model quality"
          subtitle="FLOPs linear in k · quality saturates"
          xs={series.xs}
          lines={[
            { name: 'FLOPs (× base)', color: '#fb7185', values: series.flops, max: NUM_EXPERTS, strokeDash: [] },
            { name: 'quality (0..1)', color: '#4ade80', values: series.quality.map((q) => q * NUM_EXPERTS), max: NUM_EXPERTS, strokeDash: [] },
          ]}
          cursor={k}
          highlight={2}
          highlightLabel="k = 2 · sweet spot"
          yLabel="relative"
          ticks={[1, 2, 4, 8]}
        />

        {/* Plot 2: Entropy */}
        <Plot
          title="routing entropy − Σ p log₂ p"
          subtitle="more slots → wider distributions"
          xs={series.xs}
          lines={[
            { name: 'H (bits)', color: '#67e8f9', values: series.entropy, max: 3.3, strokeDash: [] },
          ]}
          cursor={k}
          highlight={null}
          highlightLabel=""
          yLabel="bits"
          ticks={[0, 1, 2, 3]}
        />
      </div>

      <div className="absolute bottom-2 left-4 right-4 pointer-events-none flex items-center justify-between font-mono text-[10px] text-dark-text-muted">
        <span>
          <span className="text-term-green">+{qualityGain.toFixed(1)}%</span> quality
          <span className="text-dark-text-disabled"> for </span>
          <span className="text-term-rose">+{flopsCost.toFixed(0)}%</span> FLOPs
          <span className="text-dark-text-disabled"> going k=1 → 2</span>
        </span>
        <span className="text-dark-text-disabled">diminishing returns past that</span>
      </div>
    </WidgetFrame>
  )
}

function Plot({
  title,
  subtitle,
  xs,
  lines,
  cursor,
  highlight,
  highlightLabel,
  yLabel,
  ticks,
}: {
  title: string
  subtitle: string
  xs: number[]
  lines: { name: string; color: string; values: number[]; max: number; strokeDash: number[] }[]
  cursor: number
  highlight: number | null
  highlightLabel: string
  yLabel: string
  ticks: number[]
}) {
  const W = 440
  const H = 230
  const padL = 40
  const padR = 14
  const padT = 38
  const padB = 36
  const plotW = W - padL - padR
  const plotH = H - padT - padB
  const xMin = 1
  const xMax = Math.max(...xs)
  const yMax = Math.max(...lines.map((l) => l.max))

  const toSx = (x: number) => padL + ((x - xMin) / (xMax - xMin)) * plotW
  const toSy = (y: number) => padT + plotH - (y / yMax) * plotH

  return (
    <div className="flex flex-col gap-1 min-h-0">
      <div className="flex items-baseline justify-between font-mono">
        <div className="text-[11px] text-dark-text-primary">{title}</div>
        <div className="text-[9.5px] text-dark-text-muted">{subtitle}</div>
      </div>
      <div className="flex-1 min-h-0">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="w-full h-full">
          {/* Y gridlines */}
          {ticks.map((t) => (
            <g key={t}>
              <line x1={padL} y1={toSy(t)} x2={W - padR} y2={toSy(t)} stroke="#1e1e1e" strokeWidth={1} />
              <text
                x={padL - 6}
                y={toSy(t) + 3}
                textAnchor="end"
                fontSize="9"
                fill="#555"
                fontFamily="JetBrains Mono, monospace"
              >
                {t}
              </text>
            </g>
          ))}

          {/* X ticks */}
          {xs.map((x) => (
            <text
              key={x}
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
            k (experts per token)
          </text>
          <text
            x={10}
            y={padT + 4}
            fontSize="9"
            fill="#777"
            fontFamily="JetBrains Mono, monospace"
          >
            {yLabel}
          </text>

          {/* Highlight region (e.g. k=2) */}
          {highlight !== null && (
            <g>
              <rect
                x={toSx(highlight) - 14}
                y={padT}
                width={28}
                height={plotH}
                fill="#fbbf24"
                opacity={0.08}
              />
              <line
                x1={toSx(highlight)}
                y1={padT}
                x2={toSx(highlight)}
                y2={padT + plotH}
                stroke="#fbbf24"
                strokeWidth={1}
                opacity={0.5}
                strokeDasharray="3 3"
              />
              <text
                x={toSx(highlight)}
                y={padT - 20}
                textAnchor="middle"
                fontSize="9.5"
                fill="#fbbf24"
                fontFamily="JetBrains Mono, monospace"
              >
                {highlightLabel}
              </text>
            </g>
          )}

          {/* Lines */}
          {lines.map((l) => {
            const path = l.values
              .map((v, i) => `${i === 0 ? 'M' : 'L'} ${toSx(xs[i]).toFixed(1)} ${toSy(v).toFixed(1)}`)
              .join(' ')
            return (
              <g key={l.name}>
                <path
                  d={path}
                  stroke={l.color}
                  strokeWidth={2}
                  fill="none"
                  strokeDasharray={l.strokeDash.length ? l.strokeDash.join(' ') : undefined}
                />
                {xs.map((x, i) => (
                  <circle
                    key={`${l.name}-${i}`}
                    cx={toSx(x)}
                    cy={toSy(l.values[i])}
                    r={2}
                    fill={l.color}
                  />
                ))}
              </g>
            )
          })}

          {/* Cursor */}
          <line
            x1={toSx(cursor)}
            y1={padT}
            x2={toSx(cursor)}
            y2={padT + plotH}
            stroke="rgba(255,255,255,0.3)"
            strokeWidth={1}
            strokeDasharray="2 3"
          />
          {lines.map((l) => {
            const idx = xs.findIndex((x) => x === cursor)
            if (idx < 0) return null
            return (
              <circle
                key={`cursor-${l.name}`}
                cx={toSx(cursor)}
                cy={toSy(l.values[idx])}
                r={4}
                fill={l.color}
                stroke="#0a0a0a"
                strokeWidth={1}
              />
            )
          })}

          {/* Legend */}
          <g transform={`translate(${padL + 8}, ${padT - 2})`}>
            {lines.map((l, i) => (
              <g key={l.name} transform={`translate(${i * 130}, 0)`}>
                <line x1={0} y1={0} x2={14} y2={0} stroke={l.color} strokeWidth={2} />
                <text x={18} y={3} fontSize="9.5" fill="#ccc" fontFamily="JetBrains Mono, monospace">
                  {l.name}
                </text>
              </g>
            ))}
          </g>
        </svg>
      </div>
    </div>
  )
}
