'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// An MoE layer visualized as a router + 8 expert MLPs in parallel. The user
// clicks one of four tokens; the router scores all 8 experts, picks the
// top-2, and lights up the forward path — router → two experts → weighted
// sum → output. The six unused experts stay dark on purpose: those are the
// FLOPs we aren't spending.

const NUM_EXPERTS = 8
const K = 2

// Pre-baked router affinities per token so each token has a distinct personality.
// The numbers are logits; softmax + top-2 happens at render time.
const TOKENS: { label: string; caption: string; logits: number[] }[] = [
  {
    label: '"the"',
    caption: 'common function word — peaks at generalist experts',
    logits: [2.4, 1.1, 0.4, 2.9, -0.1, 0.6, 1.0, 0.3],
  },
  {
    label: '"photon"',
    caption: 'science-y lexical item — picks specialized experts',
    logits: [0.1, 0.4, 3.1, 0.7, 2.6, -0.3, 0.2, 0.9],
  },
  {
    label: '"else:"',
    caption: 'code-shaped token — routes to coding experts',
    logits: [-0.2, 2.8, 0.3, 0.1, 0.7, 3.2, 0.4, 0.0],
  },
  {
    label: '"森"',
    caption: 'CJK ideograph — routes to multilingual experts',
    logits: [0.5, 0.2, 1.1, 0.4, 0.6, 0.8, 3.1, 2.7],
  },
]

function softmax(xs: number[]): number[] {
  const m = Math.max(...xs)
  const es = xs.map((x) => Math.exp(x - m))
  const s = es.reduce((a, b) => a + b, 0)
  return es.map((e) => e / s)
}

function topK(xs: number[], k: number): number[] {
  return xs
    .map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v)
    .slice(0, k)
    .map((o) => o.i)
}

// SVG layout constants
const VB_W = 960
const VB_H = 360
const TOKEN_X = 60
const ROUTER_X = 200
const ROUTER_W = 110
const EXPERT_X = 430
const EXPERT_W = 130
const OUT_X = 720

export default function MoEBlockDiagram() {
  const [tokIdx, setTokIdx] = useState(0)
  const tok = TOKENS[tokIdx]

  const { routerProbs, selected, weights, output } = useMemo(() => {
    const rp = softmax(tok.logits)
    const sel = topK(rp, K)
    const rawW = sel.map((i) => rp[i])
    const sum = rawW.reduce((a, b) => a + b, 0)
    const w = rawW.map((x) => x / sum)
    return {
      routerProbs: rp,
      selected: sel,
      weights: w,
      output: w.reduce((a, b) => a + b, 0), // always 1, just a readout
    }
  }, [tok])

  const isActive = (e: number) => selected.includes(e)
  const weightOf = (e: number) => weights[selected.indexOf(e)] ?? 0

  // Expert vertical layout
  const gap = 14
  const totalH = NUM_EXPERTS * 34 + (NUM_EXPERTS - 1) * gap
  const startY = (VB_H - totalH) / 2

  const expertCY = (e: number) => startY + e * (34 + gap) + 17

  return (
    <WidgetFrame
      widgetName="MoEBlockDiagram"
      label="MoE layer — router lights up the two experts that will run"
      right={<span className="font-mono">k = 2 of 8 · top-2 gating</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5 flex-wrap">
            {TOKENS.map((t, i) => (
              <button
                key={t.label}
                onClick={() => setTokIdx(i)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
                  tokIdx === i
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {t.label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="active"
              value={`E${selected[0]}, E${selected[1]}`}
              accent="text-term-amber"
            />
            <Readout
              label="FLOPs"
              value={`${K} / ${NUM_EXPERTS} experts`}
              accent="text-term-green"
            />
            <Readout label="Σ weights" value={output.toFixed(3)} accent="text-term-cyan" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full h-full">
          <defs>
            <marker id="moe-arrow-amber" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="#fbbf24" />
            </marker>
            <marker id="moe-arrow-dim" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="#2a2a32" />
            </marker>
          </defs>

          {/* Token box */}
          <rect
            x={TOKEN_X}
            y={VB_H / 2 - 32}
            width={100}
            height={64}
            rx={8}
            fill="#1a1a28"
            stroke="#a78bfa"
            strokeWidth={1.4}
          />
          <text
            x={TOKEN_X + 50}
            y={VB_H / 2 - 10}
            textAnchor="middle"
            fontSize="10"
            fill="#a78bfa"
            fontFamily="JetBrains Mono, monospace"
          >
            token
          </text>
          <text
            x={TOKEN_X + 50}
            y={VB_H / 2 + 10}
            textAnchor="middle"
            fontSize="13"
            fill="#e5e7eb"
            fontFamily="JetBrains Mono, monospace"
            fontWeight={600}
          >
            {tok.label}
          </text>

          {/* Token → router wire */}
          <line
            x1={TOKEN_X + 100}
            y1={VB_H / 2}
            x2={ROUTER_X}
            y2={VB_H / 2}
            stroke="#a78bfa"
            strokeWidth={1.6}
            markerEnd="url(#moe-arrow-amber)"
          />

          {/* Router box with 8 mini-bars */}
          <rect
            x={ROUTER_X}
            y={VB_H / 2 - 100}
            width={ROUTER_W}
            height={200}
            rx={8}
            fill="#0f0f1a"
            stroke="#67e8f9"
            strokeWidth={1.4}
          />
          <text
            x={ROUTER_X + ROUTER_W / 2}
            y={VB_H / 2 - 85}
            textAnchor="middle"
            fontSize="10"
            fill="#67e8f9"
            fontFamily="JetBrains Mono, monospace"
          >
            router
          </text>
          <text
            x={ROUTER_X + ROUTER_W / 2}
            y={VB_H / 2 - 72}
            textAnchor="middle"
            fontSize="8.5"
            fill="#6b7280"
            fontFamily="JetBrains Mono, monospace"
          >
            softmax · top-2
          </text>

          {/* Router logit bars */}
          {routerProbs.map((p, i) => {
            const y = VB_H / 2 - 55 + i * 17
            const w = (ROUTER_W - 18) * p * 1.8 // scale for visibility
            const active = isActive(i)
            return (
              <g key={`bar-${i}`}>
                <rect
                  x={ROUTER_X + 6}
                  y={y}
                  width={ROUTER_W - 12}
                  height={11}
                  rx={2}
                  fill="#1a1a28"
                />
                <rect
                  x={ROUTER_X + 6}
                  y={y}
                  width={Math.min(ROUTER_W - 12, Math.max(2, w))}
                  height={11}
                  rx={2}
                  fill={active ? '#fbbf24' : '#2a3a4a'}
                  opacity={active ? 0.95 : 0.5}
                />
                <text
                  x={ROUTER_X + 10}
                  y={y + 8.5}
                  fontSize="8.5"
                  fill={active ? '#0a0a0a' : '#888'}
                  fontFamily="JetBrains Mono, monospace"
                  fontWeight={active ? 700 : 400}
                >
                  E{i}
                </text>
              </g>
            )
          })}

          {/* Router → each expert wire */}
          {Array.from({ length: NUM_EXPERTS }).map((_, e) => {
            const active = isActive(e)
            return (
              <line
                key={`wire-${e}`}
                x1={ROUTER_X + ROUTER_W}
                y1={VB_H / 2}
                x2={EXPERT_X}
                y2={expertCY(e)}
                stroke={active ? '#fbbf24' : '#2a2a32'}
                strokeWidth={active ? 2 : 0.8}
                opacity={active ? 0.9 : 0.5}
                markerEnd={active ? 'url(#moe-arrow-amber)' : 'url(#moe-arrow-dim)'}
              />
            )
          })}

          {/* Expert boxes */}
          {Array.from({ length: NUM_EXPERTS }).map((_, e) => {
            const active = isActive(e)
            const cy = expertCY(e)
            return (
              <g key={`expert-${e}`}>
                <rect
                  x={EXPERT_X}
                  y={cy - 17}
                  width={EXPERT_W}
                  height={34}
                  rx={6}
                  fill={active ? '#1a1a28' : '#0e0e18'}
                  stroke={active ? '#fbbf24' : '#333'}
                  strokeWidth={active ? 1.8 : 1}
                  opacity={active ? 1 : 0.5}
                />
                <text
                  x={EXPERT_X + 12}
                  y={cy + 4}
                  fontSize="10.5"
                  fill={active ? '#fbbf24' : '#6b7280'}
                  fontFamily="JetBrains Mono, monospace"
                  fontWeight={active ? 700 : 400}
                >
                  expert E{e}
                </text>
                <text
                  x={EXPERT_X + EXPERT_W - 10}
                  y={cy + 4}
                  textAnchor="end"
                  fontSize="9"
                  fill={active ? '#a7f3d0' : '#444'}
                  fontFamily="JetBrains Mono, monospace"
                >
                  MLP
                </text>
                {active && (
                  <text
                    x={EXPERT_X + EXPERT_W - 10}
                    y={cy - 22}
                    textAnchor="end"
                    fontSize="9.5"
                    fill="#fbbf24"
                    fontFamily="JetBrains Mono, monospace"
                  >
                    w = {weightOf(e).toFixed(2)}
                  </text>
                )}
              </g>
            )
          })}

          {/* Expert → sum wires (only active) */}
          {selected.map((e) => (
            <line
              key={`out-${e}`}
              x1={EXPERT_X + EXPERT_W}
              y1={expertCY(e)}
              x2={OUT_X}
              y2={VB_H / 2}
              stroke="#fbbf24"
              strokeWidth={2}
              opacity={0.9}
              markerEnd="url(#moe-arrow-amber)"
            />
          ))}

          {/* Weighted-sum node */}
          <circle cx={OUT_X} cy={VB_H / 2} r={22} fill="#1a1a28" stroke="#4ade80" strokeWidth={1.6} />
          <text
            x={OUT_X}
            y={VB_H / 2 + 4}
            textAnchor="middle"
            fontSize="14"
            fill="#4ade80"
            fontFamily="JetBrains Mono, monospace"
            fontWeight={700}
          >
            Σ
          </text>

          {/* Sum → output wire */}
          <line
            x1={OUT_X + 22}
            y1={VB_H / 2}
            x2={OUT_X + 130}
            y2={VB_H / 2}
            stroke="#4ade80"
            strokeWidth={1.6}
            markerEnd="url(#moe-arrow-amber)"
          />
          <rect x={OUT_X + 80} y={VB_H / 2 - 22} width={80} height={44} rx={6} fill="#0a0f0a" stroke="#4ade80" strokeWidth={1.2} opacity={0.2} />
          <text
            x={OUT_X + 120}
            y={VB_H / 2 - 5}
            textAnchor="middle"
            fontSize="9"
            fill="#4ade80"
            fontFamily="JetBrains Mono, monospace"
          >
            output
          </text>
          <text
            x={OUT_X + 120}
            y={VB_H / 2 + 10}
            textAnchor="middle"
            fontSize="9"
            fill="#6b7280"
            fontFamily="JetBrains Mono, monospace"
          >
            y = Σ w_i · E_i(x)
          </text>

          {/* Caption */}
          <text
            x={VB_W / 2}
            y={VB_H - 8}
            textAnchor="middle"
            fontSize="10"
            fill="#8b8b99"
            fontFamily="JetBrains Mono, monospace"
          >
            {tok.caption} — 6 of 8 experts stay dark, that&apos;s the FLOPs we saved
          </text>
        </svg>
      </div>
    </WidgetFrame>
  )
}
