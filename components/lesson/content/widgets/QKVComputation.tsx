'use client'

import { useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Step-through self-attention for the tiny sequence ["the", "cat", "sat"]. Each
// button reveals the next stage of the computation: Q = XW_Q, K = XW_K,
// V = XW_V, then the score matrix Q·Kᵀ/√d, the softmax, and finally the
// attended output Σ α·V. Weights are hand-picked (small, round) so the numbers
// are readable — this is a "worked example" more than a simulator.

const TOKENS = ['the', 'cat', 'sat'] as const

// d_model = 4, head_dim = 3. Each token gets a fixed 4-d embedding.
const X: number[][] = [
  [1.0, 0.0, 0.5, 0.2], // the
  [0.2, 1.0, 0.1, 0.3], // cat
  [0.3, 0.4, 1.0, 0.1], // sat
]

const W_Q: number[][] = [
  [0.5, 0.1, -0.2],
  [0.3, 0.7, 0.1],
  [-0.1, 0.2, 0.6],
  [0.4, -0.3, 0.2],
]

const W_K: number[][] = [
  [0.2, 0.5, 0.1],
  [0.6, 0.1, -0.2],
  [0.1, -0.3, 0.7],
  [-0.2, 0.4, 0.3],
]

const W_V: number[][] = [
  [0.4, -0.1, 0.2],
  [0.1, 0.5, -0.3],
  [0.3, 0.2, 0.6],
  [-0.2, 0.4, 0.1],
]

const HEAD_DIM = 3

function matmul(a: number[][], b: number[][]): number[][] {
  const out: number[][] = []
  for (let i = 0; i < a.length; i++) {
    const row: number[] = []
    for (let j = 0; j < b[0].length; j++) {
      let s = 0
      for (let k = 0; k < b.length; k++) s += a[i][k] * b[k][j]
      row.push(s)
    }
    out.push(row)
  }
  return out
}

function transpose(m: number[][]): number[][] {
  const out: number[][] = []
  for (let j = 0; j < m[0].length; j++) {
    const row: number[] = []
    for (let i = 0; i < m.length; i++) row.push(m[i][j])
    out.push(row)
  }
  return out
}

function softmaxRow(row: number[]): number[] {
  const m = Math.max(...row)
  const e = row.map((v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

const Q = matmul(X, W_Q)
const K = matmul(X, W_K)
const V = matmul(X, W_V)
const scores = matmul(Q, transpose(K)).map((r) => r.map((v) => v / Math.sqrt(HEAD_DIM)))
const attn = scores.map(softmaxRow)
const out = matmul(attn, V)

type Stage = 0 | 1 | 2 | 3 | 4 | 5 | 6

const STAGES: { key: Stage; label: string; caption: string }[] = [
  { key: 0, label: 'input X', caption: '3 tokens × d_model=4 — hand-picked embeddings.' },
  { key: 1, label: 'compute Q', caption: 'Q = X · W_Q — each row is a query vector (d=3).' },
  { key: 2, label: 'compute K', caption: 'K = X · W_K — each row is a key vector.' },
  { key: 3, label: 'compute V', caption: 'V = X · W_V — each row is a value vector.' },
  { key: 4, label: 'Q·Kᵀ / √d', caption: 'scores[i,j] = how much token i queries token j.' },
  { key: 5, label: 'softmax', caption: 'normalize each row — attention weights, sum to 1.' },
  { key: 6, label: 'attend', caption: 'output[i] = Σ_j attn[i,j] · V[j]. contextualized vectors.' },
]

export default function QKVComputation() {
  const [stage, setStage] = useState<Stage>(0)
  const info = STAGES[stage]

  return (
    <WidgetFrame
      widgetName="QKVComputation"
      label="self-attention computed by hand — click through each stage"
      right={<span className="font-mono">d_model = 4 · head_dim = 3 · {TOKENS.length} tokens</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-2">
          {STAGES.slice(1).map((s) => (
            <Button
              key={s.key}
              onClick={() => setStage(s.key)}
              variant={stage === s.key ? 'primary' : 'ghost'}
              disabled={s.key > stage + 1}
            >
              {s.label}
            </Button>
          ))}
          <Button onClick={() => setStage(0)} variant="ghost">reset</Button>
          <div className="ml-auto">
            <Readout label="stage" value={`${stage}/${STAGES.length - 1}`} accent="text-term-purple" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-[1fr] grid-rows-[auto_1fr_auto] gap-2">
        <div className="text-[11px] font-mono text-dark-text-muted">
          <span className="text-term-purple uppercase tracking-wider text-[10px] mr-2">{info.label}</span>
          {info.caption}
        </div>

        <div className="flex items-center justify-center gap-4 overflow-auto min-h-0">
          <Matrix
            title="X (3×4)"
            rows={TOKENS as unknown as string[]}
            cols={['d0', 'd1', 'd2', 'd3']}
            data={X}
            color="#67e8f9"
            active={stage >= 0}
          />
          <Op symbol="·" active={stage >= 1} />
          <Matrix
            title="W_Q (4×3)"
            rows={['d0', 'd1', 'd2', 'd3']}
            cols={['q0', 'q1', 'q2']}
            data={W_Q}
            color="#a78bfa"
            dimmed
            active={stage >= 1}
          />
          <Op symbol="=" active={stage >= 1} />
          <Matrix
            title="Q (3×3)"
            rows={TOKENS as unknown as string[]}
            cols={['q0', 'q1', 'q2']}
            data={Q}
            color="#a78bfa"
            active={stage >= 1}
            hidden={stage < 1}
          />
          {stage >= 2 && (
            <Matrix
              title="K (3×3)"
              rows={TOKENS as unknown as string[]}
              cols={['k0', 'k1', 'k2']}
              data={K}
              color="#f472b6"
              active
            />
          )}
          {stage >= 3 && (
            <Matrix
              title="V (3×3)"
              rows={TOKENS as unknown as string[]}
              cols={['v0', 'v1', 'v2']}
              data={V}
              color="#4ade80"
              active
            />
          )}
        </div>

        <div className="flex items-start justify-center gap-6 min-h-[150px]">
          {stage >= 4 && (
            <Matrix
              title="Q·Kᵀ / √d"
              rows={TOKENS as unknown as string[]}
              cols={TOKENS as unknown as string[]}
              data={scores}
              color="#fbbf24"
              active
              heatmap
            />
          )}
          {stage >= 5 && <SoftmaxBars attn={attn} />}
          {stage >= 6 && (
            <Matrix
              title="output (3×3)"
              rows={TOKENS as unknown as string[]}
              cols={['o0', 'o1', 'o2']}
              data={out}
              color="#67e8f9"
              active
              highlight
            />
          )}
        </div>
      </div>
    </WidgetFrame>
  )
}

function Matrix({
  title,
  rows,
  cols,
  data,
  color,
  active = true,
  dimmed = false,
  hidden = false,
  heatmap = false,
  highlight = false,
}: {
  title: string
  rows: string[]
  cols: string[]
  data: number[][]
  color: string
  active?: boolean
  dimmed?: boolean
  hidden?: boolean
  heatmap?: boolean
  highlight?: boolean
}) {
  if (hidden) return <div className="w-[120px] opacity-20" />
  const absMax = Math.max(0.01, ...data.flat().map((v) => Math.abs(v)))
  return (
    <div
      className={cn(
        'flex flex-col gap-1 rounded p-2 transition-opacity',
        dimmed ? 'opacity-50' : 'opacity-100',
        active ? '' : 'opacity-30',
        highlight && 'ring-1 ring-term-cyan/60',
      )}
    >
      <div className="text-[9px] font-mono uppercase tracking-wider" style={{ color }}>
        {title}
      </div>
      <div className="flex">
        <div className="flex flex-col justify-around mr-1 text-[9px] font-mono text-dark-text-disabled">
          {rows.map((r) => (
            <div key={r} className="h-5 flex items-center">{r}</div>
          ))}
        </div>
        <div>
          <div className="flex mb-0.5 text-[8px] font-mono text-dark-text-disabled">
            {cols.map((c) => (
              <div key={c} className="w-10 text-center">{c}</div>
            ))}
          </div>
          <div className="flex flex-col gap-[1px]">
            {data.map((row, i) => (
              <div key={i} className="flex gap-[1px]">
                {row.map((v, j) => {
                  const intensity = Math.min(1, Math.abs(v) / absMax)
                  const bg = heatmap
                    ? `rgba(251, 191, 36, ${0.08 + intensity * 0.75})`
                    : `${color}${Math.round((0.1 + intensity * 0.5) * 255).toString(16).padStart(2, '0')}`
                  return (
                    <div
                      key={j}
                      className="w-10 h-5 flex items-center justify-center font-mono text-[9.5px] tabular-nums rounded-[2px]"
                      style={{ backgroundColor: bg, color: '#e5e7eb' }}
                    >
                      {v.toFixed(2)}
                    </div>
                  )
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function Op({ symbol, active }: { symbol: string; active: boolean }) {
  return (
    <div
      className={cn(
        'text-[18px] font-mono text-dark-text-muted transition-opacity',
        active ? 'opacity-100' : 'opacity-20',
      )}
    >
      {symbol}
    </div>
  )
}

function SoftmaxBars({ attn }: { attn: number[][] }) {
  return (
    <div className="flex flex-col gap-1 rounded p-2">
      <div className="text-[9px] font-mono uppercase tracking-wider text-term-amber">
        softmax(scores) — attention weights
      </div>
      <div className="flex flex-col gap-1.5">
        {attn.map((row, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-[9.5px] font-mono text-dark-text-secondary w-8">{TOKENS[i]}</span>
            <div className="flex gap-0.5">
              {row.map((v, j) => (
                <div key={j} className="flex flex-col items-center">
                  <div className="w-14 h-8 bg-dark-surface-elevated/40 rounded-sm relative overflow-hidden">
                    <div
                      className="absolute bottom-0 left-0 right-0"
                      style={{
                        height: `${v * 100}%`,
                        backgroundColor: 'rgba(251, 191, 36, 0.7)',
                      }}
                    />
                    <div className="absolute inset-0 flex items-center justify-center font-mono text-[9px] text-dark-text-primary tabular-nums">
                      {v.toFixed(2)}
                    </div>
                  </div>
                  <div className="text-[8px] font-mono text-dark-text-disabled mt-0.5">{TOKENS[j]}</div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
