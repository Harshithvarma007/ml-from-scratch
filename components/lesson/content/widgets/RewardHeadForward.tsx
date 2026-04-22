'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Reward model forward pass: tokens → transformer stack → per-token hiddens
// → pooling (last-token / mean / attention-pool) → linear head → scalar reward.
// Click a stage to highlight it; dummy numeric values flow through. Pooling
// slider swaps the strategy and re-computes the output.

type Pool = 'last' | 'mean' | 'attn'

const TOKENS = ['The', ' quick', ' brown', ' fox', ' jumps', '.']
// Pretend per-token hidden "summary" scalars (e.g., projection of H onto reward direction).
const HIDDENS = [0.2, 0.5, 0.3, 1.1, 0.8, 0.6]
const ATTN = [0.05, 0.1, 0.1, 0.35, 0.3, 0.1]

function pool(strategy: Pool): { weights: number[]; value: number } {
  if (strategy === 'last') {
    const w = HIDDENS.map((_, i) => (i === HIDDENS.length - 1 ? 1 : 0))
    return { weights: w, value: HIDDENS[HIDDENS.length - 1] }
  }
  if (strategy === 'mean') {
    const w = HIDDENS.map(() => 1 / HIDDENS.length)
    const v = HIDDENS.reduce((a, b) => a + b, 0) / HIDDENS.length
    return { weights: w, value: v }
  }
  const w = ATTN
  const v = HIDDENS.reduce((a, b, i) => a + b * w[i], 0)
  return { weights: w, value: v }
}

type Stage = 'tokens' | 'transformer' | 'pool' | 'head'

export default function RewardHeadForward() {
  const [strat, setStrat] = useState<Pool>('last')
  const [stage, setStage] = useState<Stage>('head')

  const pooled = useMemo(() => pool(strat), [strat])
  const W_HEAD = 1.7
  const B_HEAD = -0.2
  const reward = pooled.value * W_HEAD + B_HEAD

  return (
    <WidgetFrame
      widgetName="RewardHeadForward"
      label="reward head forward pass"
      right={<span className="font-mono">click a stage to inspect it</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            {(['last', 'mean', 'attn'] as Pool[]).map((s) => (
              <button
                key={s}
                onClick={() => setStrat(s)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
                  strat === s
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {s === 'last' ? 'last-token' : s === 'mean' ? 'mean-pool' : 'attention-pool'}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="pooled" value={pooled.value.toFixed(3)} accent="text-term-cyan" />
            <Readout label="reward r" value={reward.toFixed(3)} accent={reward > 0 ? 'text-term-green' : 'text-term-rose'} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <svg viewBox="0 0 860 380" className="w-full h-full">
          {/* Stage 1: tokens */}
          <StageBox
            x={20}
            y={40}
            w={180}
            h={80}
            title="tokens"
            active={stage === 'tokens'}
            onClick={() => setStage('tokens')}
            color="#67e8f9"
          />
          {TOKENS.map((t, i) => (
            <text
              key={i}
              x={30 + (i % 3) * 55}
              y={75 + Math.floor(i / 3) * 18}
              fontSize="10"
              fontFamily="JetBrains Mono, monospace"
              fill={stage === 'tokens' ? '#67e8f9' : '#a1a1aa'}
            >
              {t.trim() || '·'}
            </text>
          ))}

          <Arrow x1={200} y1={80} x2={240} y2={80} />

          {/* Stage 2: transformer */}
          <StageBox
            x={240}
            y={40}
            w={180}
            h={80}
            title="transformer"
            active={stage === 'transformer'}
            onClick={() => setStage('transformer')}
            color="#a78bfa"
          />
          {HIDDENS.map((h, i) => (
            <g key={i}>
              <rect
                x={252 + i * 26}
                y={70}
                width={22}
                height={30}
                rx={2}
                fill={stage === 'transformer' ? '#a78bfa' : '#3f3f46'}
                opacity={0.35 + h * 0.4}
              />
              <text
                x={263 + i * 26}
                y={112}
                fontSize="8"
                fontFamily="JetBrains Mono, monospace"
                textAnchor="middle"
                fill="#888"
              >
                {h.toFixed(1)}
              </text>
            </g>
          ))}

          <Arrow x1={420} y1={80} x2={460} y2={80} />

          {/* Stage 3: pool */}
          <StageBox
            x={460}
            y={40}
            w={180}
            h={80}
            title={`pool: ${strat}`}
            active={stage === 'pool'}
            onClick={() => setStage('pool')}
            color="#fbbf24"
          />
          {pooled.weights.map((w, i) => (
            <g key={i}>
              <rect
                x={472 + i * 26}
                y={100 - w * 40}
                width={22}
                height={Math.max(2, w * 40)}
                fill="#fbbf24"
                opacity={0.7}
              />
              <text
                x={483 + i * 26}
                y={112}
                fontSize="8"
                fontFamily="JetBrains Mono, monospace"
                textAnchor="middle"
                fill="#888"
              >
                {w.toFixed(2)}
              </text>
            </g>
          ))}

          <Arrow x1={640} y1={80} x2={680} y2={80} />

          {/* Stage 4: head */}
          <StageBox
            x={680}
            y={40}
            w={160}
            h={80}
            title="linear head"
            active={stage === 'head'}
            onClick={() => setStage('head')}
            color="#4ade80"
          />
          <text x={760} y={80} textAnchor="middle" fontSize="11" fontFamily="JetBrains Mono, monospace" fill="#4ade80">
            r = {reward.toFixed(3)}
          </text>
          <text x={760} y={100} textAnchor="middle" fontSize="9" fontFamily="JetBrains Mono, monospace" fill="#888">
            W · h + b
          </text>

          {/* Reward bar */}
          <g transform="translate(20, 170)">
            <text x={0} y={-6} fontSize="10" fontFamily="JetBrains Mono, monospace" fill="#666">
              SCALAR REWARD OUTPUT
            </text>
            <rect x={0} y={0} width={820} height={22} rx={4} fill="#141420" stroke="#3f3f46" />
            <line x1={410} y1={0} x2={410} y2={22} stroke="#3f3f46" strokeDasharray="2 3" />
            <rect
              x={reward >= 0 ? 410 : 410 - Math.min(1, Math.abs(reward) / 3) * 410}
              y={2}
              width={Math.min(1, Math.abs(reward) / 3) * 410}
              height={18}
              fill={reward > 0 ? '#4ade80' : '#f87171'}
              opacity={0.75}
              rx={2}
            />
            <text x={410} y={40} textAnchor="middle" fontSize="9" fontFamily="JetBrains Mono, monospace" fill="#666">0</text>
            <text x={0} y={40} fontSize="9" fontFamily="JetBrains Mono, monospace" fill="#666">−3</text>
            <text x={820} y={40} textAnchor="end" fontSize="9" fontFamily="JetBrains Mono, monospace" fill="#666">+3</text>
          </g>

          {/* Inspection panel */}
          <g transform="translate(20, 235)">
            <rect x={0} y={0} width={820} height={125} rx={6} fill="#0f0f1a" stroke="#2a2a32" />
            <text x={14} y={22} fontSize="11" fontFamily="JetBrains Mono, monospace" fill="#fbbf24" fontWeight={600}>
              stage: {stage}
            </text>
            {stage === 'tokens' && (
              <g>
                <text x={14} y={44} fontSize="11" fontFamily="JetBrains Mono, monospace" fill="#e5e7eb">
                  input: {TOKENS.join('')!} ({TOKENS.length} tokens)
                </text>
                <text x={14} y={64} fontSize="10.5" fontFamily="JetBrains Mono, monospace" fill="#a1a1aa">
                  fed through embeddings then the transformer stack. The RM sees prompt + response — it scores the full sequence.
                </text>
              </g>
            )}
            {stage === 'transformer' && (
              <g>
                <text x={14} y={44} fontSize="11" fontFamily="JetBrains Mono, monospace" fill="#e5e7eb">
                  output h_i per position, shape [T, d]. Shown: a projection to one scalar per token for visualization.
                </text>
                <text x={14} y={64} fontSize="10.5" fontFamily="JetBrains Mono, monospace" fill="#a1a1aa">
                  The last token (jumps.) carries a high signal — response evaluations cluster in its final state.
                </text>
              </g>
            )}
            {stage === 'pool' && (
              <g>
                <text x={14} y={44} fontSize="11" fontFamily="JetBrains Mono, monospace" fill="#e5e7eb">
                  {strat === 'last' && 'last-token: h_pool = h_T. Simple, dominant in practice for causal LMs.'}
                  {strat === 'mean' && 'mean-pool: h_pool = (1/T) · Σ h_i. Democratic; every token weighs in.'}
                  {strat === 'attn' && 'attention-pool: h_pool = Σ α_i h_i, α from a small learned query.'}
                </text>
                <text x={14} y={66} fontSize="10.5" fontFamily="JetBrains Mono, monospace" fill="#a1a1aa">
                  pooled value = {pooled.value.toFixed(3)}
                </text>
              </g>
            )}
            {stage === 'head' && (
              <g>
                <text x={14} y={44} fontSize="11" fontFamily="JetBrains Mono, monospace" fill="#e5e7eb">
                  reward head: a single linear layer — one scalar output r = W · h_pool + b.
                </text>
                <text x={14} y={66} fontSize="10.5" fontFamily="JetBrains Mono, monospace" fill="#a1a1aa">
                  W = {W_HEAD.toFixed(2)}, b = {B_HEAD.toFixed(2)} → r = {W_HEAD.toFixed(2)} · {pooled.value.toFixed(2)} + ({B_HEAD.toFixed(2)}) = {reward.toFixed(3)}
                </text>
                <text x={14} y={88} fontSize="10.5" fontFamily="JetBrains Mono, monospace" fill="#888">
                  Trained so chosen responses get higher r than rejected ones (Bradley-Terry).
                </text>
              </g>
            )}
          </g>
        </svg>
      </div>
    </WidgetFrame>
  )
}

function StageBox({
  x,
  y,
  w,
  h,
  title,
  active,
  onClick,
  color,
}: {
  x: number
  y: number
  w: number
  h: number
  title: string
  active: boolean
  onClick: () => void
  color: string
}) {
  return (
    <g onClick={onClick} style={{ cursor: 'pointer' }}>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={8}
        fill={active ? '#1f2937' : '#141420'}
        stroke={active ? color : '#3f3f46'}
        strokeWidth={active ? 2 : 1.2}
      />
      <text
        x={x + w / 2}
        y={y - 6}
        textAnchor="middle"
        fontSize="10"
        fontFamily="JetBrains Mono, monospace"
        fill={active ? color : '#888'}
        letterSpacing="0.5"
      >
        {title.toUpperCase()}
      </text>
    </g>
  )
}

function Arrow({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) {
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2 - 6} y2={y2} stroke="#6b7280" strokeWidth={1.4} />
      <polygon points={`${x2},${y2} ${x2 - 7},${y2 - 4} ${x2 - 7},${y2 + 4}`} fill="#6b7280" />
    </g>
  )
}
