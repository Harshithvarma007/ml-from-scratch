'use client'

import { useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Rotary Position Embedding in 2D. Pick a dim-pair index (each pair has its
// own rotation frequency) and slide the position slider — the query vector
// rotates by θ = pos · 10000^(-2i/d) at each step. The live rotation matrix
// is printed next to the plot. A multi-pane footer shows three different
// dim-pairs rotating side-by-side, each at its own frequency.

const D_MODEL = 64
const Q_BASE: readonly [number, number] = [0.9, 0.3]

function rotate(v: readonly [number, number], theta: number): [number, number] {
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  return [c * v[0] - s * v[1], s * v[0] + c * v[1]]
}

function thetaFor(i: number, pos: number): number {
  return pos * Math.pow(10000, (-2 * i) / D_MODEL)
}

export default function RoPERotation() {
  const [pos, setPos] = useState(12)
  const [pairIdx, setPairIdx] = useState(0)

  const theta = thetaFor(pairIdx, pos)
  const rotated = rotate(Q_BASE, theta)
  const period = (2 * Math.PI) / Math.max(Math.pow(10000, (-2 * pairIdx) / D_MODEL), 1e-9)
  const cos = Math.cos(theta)
  const sin = Math.sin(theta)

  // Three side-by-side pairs at contrasting frequencies (low/mid/high i).
  const triPairs: readonly number[] = [0, 8, 24]

  return (
    <WidgetFrame
      widgetName="RoPERotation"
      label="RoPE — position becomes a rotation angle"
      right={<span className="font-mono">θ_i = pos · 10000^(-2i/d) · q_rot = R(θ) · q</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="pos"
            value={pos}
            min={0}
            max={64}
            step={1}
            onChange={(v) => setPos(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-cyan"
          />
          <Slider
            label="dim-pair i"
            value={pairIdx}
            min={0}
            max={D_MODEL / 2 - 1}
            step={1}
            onChange={(v) => setPairIdx(Math.round(v))}
            format={(v) => `${Math.round(v)} / ${D_MODEL / 2 - 1}`}
            accent="accent-term-purple"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="θ" value={`${theta.toFixed(3)} rad`} accent="text-term-amber" />
            <Readout label="period" value={period > 1e5 ? '∞' : period.toFixed(1)} accent="text-term-teal" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-rows-[1fr_1fr] gap-3 overflow-hidden">
        {/* Top: main rotation + matrix */}
        <div className="grid grid-cols-1 md:grid-cols-[1fr_280px] gap-3 min-h-0">
          <div className="bg-dark-surface-elevated/20 border border-dark-border rounded relative">
            <RotationView pos={pos} pairIdx={pairIdx} theta={theta} rotated={rotated} showTrail />
          </div>

          {/* Rotation matrix readout */}
          <div className="flex flex-col gap-3 min-w-0">
            <div className="bg-dark-surface-elevated/40 border border-dark-border rounded p-3">
              <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
                R(θ) — live values
              </div>
              <svg viewBox="0 0 240 110" className="w-full">
                <text x={4} y={58} fontSize="22" fill="#666" fontFamily="JetBrains Mono, monospace">[</text>
                <text x={20} y={40} fontSize="14" fill="#67e8f9" fontFamily="JetBrains Mono, monospace">{cos.toFixed(3)}</text>
                <text x={100} y={40} fontSize="14" fill="#f87171" fontFamily="JetBrains Mono, monospace">{(-sin).toFixed(3)}</text>
                <text x={20} y={76} fontSize="14" fill="#fbbf24" fontFamily="JetBrains Mono, monospace">{sin.toFixed(3)}</text>
                <text x={100} y={76} fontSize="14" fill="#4ade80" fontFamily="JetBrains Mono, monospace">{cos.toFixed(3)}</text>
                <text x={184} y={58} fontSize="22" fill="#666" fontFamily="JetBrains Mono, monospace">]</text>
                <text x={200} y={44} fontSize="10" fill="#888" fontFamily="JetBrains Mono, monospace">cos θ</text>
                <text x={200} y={80} fontSize="10" fill="#888" fontFamily="JetBrains Mono, monospace">sin θ</text>
              </svg>
              <div className="pt-2 font-mono text-[10.5px] text-dark-text-muted border-t border-dark-border mt-2">
                <div>q = [{Q_BASE[0].toFixed(2)}, {Q_BASE[1].toFixed(2)}]</div>
                <div className="text-term-green mt-1">
                  R·q = [{rotated[0].toFixed(3)}, {rotated[1].toFixed(3)}]
                </div>
              </div>
            </div>
            <div className="bg-dark-surface-elevated/30 border border-dark-border rounded p-3 font-mono text-[10.5px] text-dark-text-muted leading-snug">
              <div className="text-dark-text-disabled uppercase tracking-wider text-[9.5px] mb-1">why this works</div>
              each dim-pair rotates at its own rate; high-i pairs rotate slowly, low-i pairs
              rotate fast. the inner product depends on{' '}
              <span className="text-term-teal">pos_q − pos_k</span> only — pure relative position.
            </div>
          </div>
        </div>

        {/* Bottom: three dim pairs rotating side by side */}
        <div className="grid grid-cols-3 gap-3 min-h-0">
          {triPairs.map((i) => {
            const t = thetaFor(i, pos)
            const per = (2 * Math.PI) / Math.max(Math.pow(10000, (-2 * i) / D_MODEL), 1e-9)
            return (
              <div
                key={i}
                className="bg-dark-surface-elevated/20 border border-dark-border rounded relative flex flex-col"
              >
                <div className="absolute top-1.5 left-2 right-2 flex justify-between font-mono text-[9.5px] z-10 pointer-events-none">
                  <span className="text-term-purple">i = {i}</span>
                  <span className="text-dark-text-disabled">
                    period ≈ {per > 1e5 ? '∞' : per.toFixed(0)}
                  </span>
                </div>
                <RotationView pos={pos} pairIdx={i} theta={t} rotated={rotate(Q_BASE, t)} compact />
              </div>
            )
          })}
        </div>
      </div>
    </WidgetFrame>
  )
}

function RotationView({
  pos,
  pairIdx,
  theta,
  rotated,
  compact,
  showTrail,
}: {
  pos: number
  pairIdx: number
  theta: number
  rotated: readonly [number, number]
  compact?: boolean
  showTrail?: boolean
}) {
  const size = compact ? 220 : 420
  const cx = size / 2
  const cy = size / 2
  const R = size * 0.36
  const scale = R // unit radius → pixel radius

  const qx = cx + Q_BASE[0] * scale
  const qy = cy - Q_BASE[1] * scale
  const rx = cx + rotated[0] * scale
  const ry = cy - rotated[1] * scale

  // Trail: draw previous positions (0..pos) to show the swept path.
  const trail: { x: number; y: number }[] = []
  if (showTrail) {
    const steps = Math.min(48, Math.max(2, pos * 2))
    for (let s = 0; s <= steps; s++) {
      const p = (s / steps) * pos
      const th = p * Math.pow(10000, (-2 * pairIdx) / D_MODEL)
      const r = rotate(Q_BASE, th)
      trail.push({ x: cx + r[0] * scale, y: cy - r[1] * scale })
    }
  }

  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="w-full h-full" preserveAspectRatio="xMidYMid meet">
      {/* Unit circle */}
      <circle cx={cx} cy={cy} r={R} fill="none" stroke="#222" strokeWidth={1} strokeDasharray="3 3" />
      {/* Axes */}
      <line x1={cx - R - 10} y1={cy} x2={cx + R + 10} y2={cy} stroke="#2a2a32" strokeWidth={1} />
      <line x1={cx} y1={cy - R - 10} x2={cx} y2={cy + R + 10} stroke="#2a2a32" strokeWidth={1} />
      <text x={cx + R + 12} y={cy + 4} fontSize="10" fill="#555" fontFamily="JetBrains Mono, monospace">x</text>
      <text x={cx - 4} y={cy - R - 12} fontSize="10" fill="#555" fontFamily="JetBrains Mono, monospace">y</text>

      {/* Trail */}
      {showTrail && trail.length > 1 && (
        <polyline
          points={trail.map((p) => `${p.x},${p.y}`).join(' ')}
          fill="none"
          stroke="rgba(74, 222, 128, 0.25)"
          strokeWidth={1.2}
        />
      )}

      {/* Angle arc */}
      {Math.abs(theta) > 0.01 && (
        <path
          d={`M ${cx + 28} ${cy} A 28 28 0 ${Math.abs(theta) > Math.PI ? 1 : 0} ${theta > 0 ? 0 : 1} ${cx + 28 * Math.cos(-theta)} ${cy + 28 * Math.sin(-theta)}`}
          fill="none"
          stroke="#fbbf24"
          strokeWidth={1.4}
        />
      )}

      {/* Base vector q */}
      <line x1={cx} y1={cy} x2={qx} y2={qy} stroke="#67e8f9" strokeWidth={1.4} strokeDasharray="4 3" />
      <circle cx={qx} cy={qy} r={3.5} fill="#67e8f9" />
      {!compact && (
        <text x={qx + 6} y={qy - 6} fontSize="10" fill="#67e8f9" fontFamily="JetBrains Mono, monospace">
          q (pos = 0)
        </text>
      )}

      {/* Rotated vector */}
      <line x1={cx} y1={cy} x2={rx} y2={ry} stroke="#4ade80" strokeWidth={2} markerEnd={`url(#rope-arrow-${compact ? 'c' : 'm'})`} />
      <circle cx={rx} cy={ry} r={4.5} fill="#4ade80" />
      {!compact && (
        <text x={rx + 8} y={ry - 4} fontSize="11" fill="#4ade80" fontFamily="JetBrains Mono, monospace">
          R(θ)·q
        </text>
      )}

      {/* θ readout in center */}
      <text
        x={cx}
        y={compact ? size - 10 : size - 14}
        textAnchor="middle"
        fontSize={compact ? 10 : 12}
        fill="#fbbf24"
        fontFamily="JetBrains Mono, monospace"
      >
        θ = {theta.toFixed(3)} rad · pos = {pos}
      </text>

      <defs>
        <marker id={`rope-arrow-${compact ? 'c' : 'm'}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#4ade80" />
        </marker>
      </defs>
    </svg>
  )
}
