'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Classifier-free guidance scale plot: as w increases from 0 to 15, CLIP
// alignment rises (saturating) and diversity/inverse-FID falls. Curves are
// analytical — alignment follows a sigmoid-like saturation, diversity a
// soft logistic decay. A vertical cursor tracks w. A small tile grid mirrors
// the effect: 4 conditional samples whose "alignment" strength scales with w.

const W_MAX = 15

function alignmentCurve(w: number): number {
  // Saturating climb: 1 - 1/(1 + 0.6*w)^0.9
  return 1 - 1 / Math.pow(1 + 0.6 * w, 0.9)
}

function diversityCurve(w: number): number {
  // Soft decay: exp(-0.22 * w) with a floor
  return 0.2 + 0.8 * Math.exp(-0.22 * w)
}

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const N_TILE = 24

function sampleTile(seed: number, alignment: number): number[] {
  const rng = mulberry32(seed)
  const out = new Array(N_TILE * N_TILE)
  // Target shape is a circle; as alignment → 1, tile looks more like the circle.
  for (let y = 0; y < N_TILE; y++) {
    for (let x = 0; x < N_TILE; x++) {
      const d = Math.hypot(x - 12, y - 12)
      const target = d < 7 ? 0.85 : -0.5
      const noise = (rng() - 0.5) * 2.0
      out[y * N_TILE + x] = alignment * target + (1 - alignment) * noise
    }
  }
  return out
}

function toGray(v: number): string {
  const g = Math.max(0, Math.min(255, Math.round((v * 0.5 + 0.5) * 255)))
  return `rgb(${g},${g},${g})`
}

export default function CFGScaleEffect() {
  const [w, setW] = useState(2)

  const align = alignmentCurve(w)
  const div = diversityCurve(w)

  // Sample four tiles that gain structure as w rises
  const tiles = useMemo(
    () => [0, 1, 2, 3].map((i) => sampleTile(i + 1, Math.min(1, align * 1.05))),
    [align],
  )

  // Precompute curve paths
  const curves = useMemo(() => {
    const W = 340
    const H = 160
    const padL = 36
    const padR = 16
    const padT = 14
    const padB = 22
    const pw = W - padL - padR
    const ph = H - padT - padB
    const toX = (x: number) => padL + (x / W_MAX) * pw
    const toY = (y: number) => padT + ph - y * ph
    let alignPath = ''
    let divPath = ''
    for (let i = 0; i <= 100; i++) {
      const x = (i / 100) * W_MAX
      const a = alignmentCurve(x)
      const d = diversityCurve(x)
      alignPath += `${i === 0 ? 'M' : 'L'} ${toX(x).toFixed(1)} ${toY(a).toFixed(1)} `
      divPath += `${i === 0 ? 'M' : 'L'} ${toX(x).toFixed(1)} ${toY(d).toFixed(1)} `
    }
    return { W, H, padL, padR, padT, padB, toX, toY, alignPath, divPath }
  }, [])

  const { W, H, padL, padR, padT, padB, toX, toY, alignPath, divPath } = curves
  const pw = W - padL - padR
  const ph = H - padT - padB

  return (
    <WidgetFrame
      widgetName="CFGScaleEffect"
      label="CFG scale w — alignment vs diversity"
      right={<span className="font-mono">ε̂ = ε_uncond + w·(ε_cond − ε_uncond)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="guidance w"
            value={w}
            min={0}
            max={W_MAX}
            step={0.1}
            onChange={setW}
            format={(v) => v.toFixed(1)}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="CLIP align" value={align.toFixed(2)} accent="text-term-green" />
            <Readout label="diversity" value={div.toFixed(2)} accent="text-term-cyan" />
            <Readout
              label="regime"
              value={w < 1 ? 'weak' : w < 5 ? 'balanced' : w < 10 ? 'strong' : 'saturated'}
              accent={w < 1 ? 'text-term-rose' : w < 5 ? 'text-term-green' : w < 10 ? 'text-term-amber' : 'text-term-rose'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-1 md:grid-cols-[1fr_220px] gap-4">
        {/* curves */}
        <div className="flex flex-col gap-2 min-w-0 min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            trade-off curves
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full max-h-[220px]">
            {/* grid */}
            {[0, 0.25, 0.5, 0.75, 1].map((y) => (
              <g key={y}>
                <line x1={padL} y1={toY(y)} x2={W - padR} y2={toY(y)} stroke="#1a1a1f" strokeWidth={0.8} />
                <text x={padL - 4} y={toY(y) + 3} fontSize="9" textAnchor="end" fill="#555" fontFamily="JetBrains Mono, monospace">
                  {y.toFixed(2)}
                </text>
              </g>
            ))}
            {[0, 5, 10, 15].map((x) => (
              <text key={x} x={toX(x)} y={H - padB + 14} fontSize="9" textAnchor="middle" fill="#555" fontFamily="JetBrains Mono, monospace">
                {x}
              </text>
            ))}
            <text x={padL + pw / 2} y={H - 2} fontSize="9.5" textAnchor="middle" fill="#777" fontFamily="JetBrains Mono, monospace">
              w (guidance scale)
            </text>

            {/* curves */}
            <path d={alignPath} stroke="#4ade80" strokeWidth={1.8} fill="none" />
            <path d={divPath} stroke="#67e8f9" strokeWidth={1.8} fill="none" />

            {/* cursor */}
            <line x1={toX(w)} y1={padT} x2={toX(w)} y2={padT + ph} stroke="rgba(251,191,36,0.6)" strokeDasharray="3 3" />
            <circle cx={toX(w)} cy={toY(align)} r={3.5} fill="#4ade80" />
            <circle cx={toX(w)} cy={toY(div)} r={3.5} fill="#67e8f9" />

            {/* legend */}
            <g transform={`translate(${padL + 10}, ${padT + 4})`}>
              <line x1={0} y1={8} x2={12} y2={8} stroke="#4ade80" strokeWidth={2} />
              <text x={16} y={11} fontSize="10" fill="#ccc" fontFamily="JetBrains Mono, monospace">alignment ↑</text>
              <line x1={108} y1={8} x2={120} y2={8} stroke="#67e8f9" strokeWidth={2} />
              <text x={124} y={11} fontSize="10" fill="#ccc" fontFamily="JetBrains Mono, monospace">diversity ↓</text>
            </g>
          </svg>

          <div className="text-[10.5px] font-mono text-dark-text-muted leading-snug">
            w = 0 → purely unconditional · w = 1 → conditional · w &gt; 1 → amplify the
            conditional direction. Beyond ~7 the tradeoff flattens and samples start to
            saturate.
          </div>
        </div>

        {/* samples */}
        <div className="flex flex-col gap-2 min-w-0 min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            samples @ w = {w.toFixed(1)}
          </div>
          <div className="grid grid-cols-2 gap-2">
            {tiles.map((tile, i) => (
              <div
                key={i}
                className="aspect-square rounded border border-dark-border overflow-hidden"
                style={{
                  display: 'grid',
                  gridTemplateColumns: `repeat(${N_TILE}, 1fr)`,
                  gridTemplateRows: `repeat(${N_TILE}, 1fr)`,
                }}
              >
                {tile.map((v, j) => (
                  <div key={j} style={{ backgroundColor: toGray(v) }} />
                ))}
              </div>
            ))}
          </div>
          <div className="text-[10px] font-mono text-dark-text-muted leading-snug">
            target: a filled circle. As w rises each tile converges toward the target — at
            the cost of all four tiles looking alike.
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
