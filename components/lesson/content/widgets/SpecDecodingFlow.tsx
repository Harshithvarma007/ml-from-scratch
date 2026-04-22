'use client'

import { useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { SkipBack, StepForward } from 'lucide-react'
import { cn } from '@/lib/utils'

// Speculative decoding timeline over multiple rounds. Draft proposes K
// tokens cheaply; target verifies all K in one parallel forward pass and
// accepts the longest matching prefix. Tokens after the first mismatch are
// thrown away; a fresh token is resampled at the mismatch position.
//
// A "step" advances one round (one target pass). We scripted three rounds
// with different accept counts so the user can see the bookkeeping.

const K = 4 // draft proposes 4 tokens per round

type RoundOutcome = {
  draft: string[]       // 4 draft tokens
  targetMatches: boolean[] // matches target
  resampled: string     // replaces the first rejected token
  commentary: string
}

// Synthetic script: accepted-accepted-rejected-discarded, resampled.
const ROUNDS: RoundOutcome[] = [
  {
    draft: ['the', 'cat', 'sat', 'on'],
    targetMatches: [true, true, true, true],
    resampled: 'the',
    commentary: 'all 4 drafts matched — bonus token generated free, net 5 tokens this round',
  },
  {
    draft: ['mat', 'in', 'the', 'sun'],
    targetMatches: [true, true, true, false],
    resampled: 'park',
    commentary: '3 accepted, 4th rejected — target resamples \u201cpark\u201d at position 4, net 4 tokens',
  },
  {
    draft: ['and', 'watched', 'the', 'birds'],
    targetMatches: [true, false, false, false],
    resampled: 'dreamed',
    commentary: 'only 1 accept — cheap draft missed the topic shift; net 2 tokens',
  },
]

type Phase = 'idle' | 'draft' | 'verify' | 'done'

function buildFrames() {
  const frames: Array<{
    round: number
    phase: Phase
    accepted: string[]
    commentary: string
    acceptsSoFar: number
  }> = []
  let accepted: string[] = []
  let acceptCount = 0
  frames.push({ round: -1, phase: 'idle', accepted: [], commentary: 'ready — press step to start round 1', acceptsSoFar: 0 })
  for (let r = 0; r < ROUNDS.length; r++) {
    frames.push({ round: r, phase: 'draft', accepted: [...accepted], commentary: `round ${r + 1}: draft proposes 4 tokens`, acceptsSoFar: acceptCount })
    frames.push({ round: r, phase: 'verify', accepted: [...accepted], commentary: `round ${r + 1}: target verifies all 4 in one parallel pass`, acceptsSoFar: acceptCount })
    const outcome = ROUNDS[r]
    const firstMiss = outcome.targetMatches.findIndex((m) => !m)
    const take = firstMiss === -1 ? K : firstMiss
    accepted = [...accepted, ...outcome.draft.slice(0, take), outcome.resampled]
    acceptCount += take + 1 // accepted prefix + resampled token
    frames.push({ round: r, phase: 'done', accepted: [...accepted], commentary: outcome.commentary, acceptsSoFar: acceptCount })
  }
  return frames
}

const FRAMES = buildFrames()

export default function SpecDecodingFlow() {
  const [i, setI] = useState(0)
  const frame = FRAMES[i]

  const totalTargetCalls = FRAMES.slice(0, i + 1).filter((f) => f.phase === 'verify' || f.phase === 'done').length
  const targetCalls = FRAMES.filter((f) => f.phase === 'verify').length
  const actualCalls = Math.max(0, Math.ceil(totalTargetCalls / 2))

  return (
    <WidgetFrame
      widgetName="SpecDecodingFlow"
      label="speculative decoding — draft proposes, target verifies in parallel"
      right={<span className="font-mono">K = {K} drafts/round · one target pass per round</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <Button onClick={() => setI(0)}>
              <span className="inline-flex items-center gap-1">
                <SkipBack size={11} /> reset
              </span>
            </Button>
            <Button
              onClick={() => setI(Math.min(FRAMES.length - 1, i + 1))}
              variant="primary"
              disabled={i === FRAMES.length - 1}
            >
              <span className="inline-flex items-center gap-1">
                step <StepForward size={11} />
              </span>
            </Button>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="committed" value={String(frame.accepted.length)} accent="text-term-green" />
            <Readout label="target calls" value={String(actualCalls)} accent="text-term-cyan" />
            <Readout
              label="tokens/call"
              value={actualCalls > 0 ? (frame.accepted.length / actualCalls).toFixed(2) : '—'}
              accent="text-term-amber"
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <div className="grid h-full grid-rows-[auto_1fr_auto] gap-3 min-h-0">
          {/* Committed tokens strip */}
          <div>
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-1">
              committed stream
            </div>
            <div className="flex flex-wrap gap-1 p-2 rounded border border-dark-border bg-dark-bg/60 min-h-[38px]">
              {frame.accepted.length === 0 ? (
                <span className="font-mono text-[10.5px] text-dark-text-disabled italic">(empty — no rounds yet)</span>
              ) : (
                frame.accepted.map((tok, k) => (
                  <span
                    key={k}
                    className="px-2 py-1 rounded font-mono text-[11px] bg-term-green/15 border border-term-green/40 text-term-green"
                  >
                    {tok}
                  </span>
                ))
              )}
            </div>
          </div>

          {/* Timeline SVG */}
          <div className="min-h-0 rounded border border-dark-border bg-dark-bg/60 overflow-hidden">
            <svg viewBox="0 0 780 260" preserveAspectRatio="none" className="w-full h-full">
              <Timeline activeRound={frame.round} phase={frame.phase} />
            </svg>
          </div>

          {/* Commentary */}
          <div className="font-mono text-[11px] text-dark-text-muted px-1">
            <span className="text-term-amber uppercase tracking-wider text-[10px] mr-2">step {i} —</span>
            {frame.commentary}
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function Timeline({ activeRound, phase }: { activeRound: number; phase: Phase }) {
  const roundWidth = 230
  const leftPad = 50
  const topPad = 20

  return (
    <g>
      {/* Lane labels */}
      <text x={10} y={60} fontSize="10" fill="#fbbf24" fontFamily="JetBrains Mono, monospace">
        draft
      </text>
      <text x={10} y={140} fontSize="10" fill="#67e8f9" fontFamily="JetBrains Mono, monospace">
        target
      </text>
      <text x={10} y={220} fontSize="10" fill="#4ade80" fontFamily="JetBrains Mono, monospace">
        commit
      </text>

      <line x1={leftPad - 5} y1={60} x2={760} y2={60} stroke="#222" strokeWidth={1} />
      <line x1={leftPad - 5} y1={140} x2={760} y2={140} stroke="#222" strokeWidth={1} />
      <line x1={leftPad - 5} y1={220} x2={760} y2={220} stroke="#222" strokeWidth={1} />

      {ROUNDS.map((r, ri) => {
        const x0 = leftPad + ri * roundWidth
        const isActive = ri === activeRound
        const isPast = activeRound > ri
        const phaseDraft = isActive ? phase === 'draft' || phase === 'verify' || phase === 'done' : isPast
        const phaseVerify = isActive ? phase === 'verify' || phase === 'done' : isPast
        const phaseDone = isActive ? phase === 'done' : isPast
        const firstMiss = r.targetMatches.findIndex((m) => !m)
        const take = firstMiss === -1 ? K : firstMiss

        return (
          <g key={ri}>
            {/* Round header */}
            <text x={x0 + roundWidth / 2 - 15} y={topPad} fontSize="10" fill={isActive ? '#fbbf24' : '#555'} fontFamily="JetBrains Mono, monospace">
              round {ri + 1}
            </text>

            {/* Draft tokens */}
            {r.draft.map((tok, ti) => {
              const cx = x0 + ti * 46 + 20
              return (
                <g key={`d-${ti}`}>
                  <rect
                    x={cx - 20}
                    y={45}
                    width={40}
                    height={28}
                    rx={4}
                    fill={phaseDraft ? '#3a2a10' : '#141420'}
                    stroke={phaseDraft ? '#fbbf24' : '#3f3f46'}
                    strokeWidth={1}
                    opacity={phaseDraft ? 1 : 0.55}
                  />
                  <text
                    x={cx}
                    y={63}
                    textAnchor="middle"
                    fontSize="10"
                    fill={phaseDraft ? '#fbbf24' : '#555'}
                    fontFamily="JetBrains Mono, monospace"
                  >
                    {tok}
                  </text>
                </g>
              )
            })}

            {/* Single target pass bar */}
            <rect
              x={x0}
              y={125}
              width={K * 46}
              height={28}
              rx={4}
              fill={phaseVerify ? '#0a2a2e' : '#141420'}
              stroke={phaseVerify ? '#67e8f9' : '#3f3f46'}
              strokeWidth={1}
              opacity={phaseVerify ? 1 : 0.55}
            />
            <text
              x={x0 + (K * 46) / 2}
              y={143}
              textAnchor="middle"
              fontSize="10"
              fill={phaseVerify ? '#67e8f9' : '#555'}
              fontFamily="JetBrains Mono, monospace"
            >
              target(x0..x{K}) — 1 parallel pass
            </text>

            {/* Commit row: accepted tokens green, rejected red, resampled purple */}
            {r.draft.map((tok, ti) => {
              const cx = x0 + ti * 46 + 20
              let color = '#555'
              let fill = '#141420'
              let stroke = '#3f3f46'
              let label = tok
              if (phaseDone) {
                if (ti < take) {
                  color = '#4ade80'
                  fill = '#0f2017'
                  stroke = '#4ade80'
                } else if (ti === take) {
                  color = '#a78bfa'
                  fill = '#1a1530'
                  stroke = '#a78bfa'
                  label = r.resampled
                } else {
                  color = '#fb7185'
                  fill = '#2a0f14'
                  stroke = '#fb7185'
                  label = tok + ' ✕'
                }
              }
              return (
                <g key={`c-${ti}`} opacity={phaseDone ? 1 : 0.4}>
                  <rect x={cx - 20} y={205} width={40} height={28} rx={4} fill={fill} stroke={stroke} strokeWidth={1} />
                  <text x={cx} y={223} textAnchor="middle" fontSize="10" fill={color} fontFamily="JetBrains Mono, monospace">
                    {label}
                  </text>
                </g>
              )
            })}

            {/* Arrow: target -> commit at decision time */}
            {phaseDone && (
              <g>
                <line
                  x1={x0 + (K * 46) / 2}
                  y1={154}
                  x2={x0 + take * 46 + 20}
                  y2={204}
                  stroke="#a78bfa"
                  strokeWidth={1.2}
                  strokeDasharray="4 3"
                />
                <text
                  x={x0 + (K * 46) / 2}
                  y={182}
                  textAnchor="middle"
                  fontSize="9"
                  fill="#a78bfa"
                  fontFamily="JetBrains Mono, monospace"
                >
                  accept {take} + resample
                </text>
              </g>
            )}
          </g>
        )
      })}
    </g>
  )
}
