'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Walk through how a router turns a single token into mixture weights:
//   1) raw router logits z_i = (w_r · x)_i for i = 1..E
//   2) softmax to get p_i = exp(z_i) / Σ exp(z_j)
//   3) mask out everything except the top-k
//   4) renormalize so the kept weights sum to 1
// The four rows show each stage as a bar chart. A button steps through five
// canned tokens with very different affinity profiles.

const NUM_EXPERTS = 8

const TOKENS: { name: string; logits: number[]; blurb: string }[] = [
  {
    name: '"the"',
    blurb: 'soft, broad affinity — almost any expert works',
    logits: [1.4, 1.6, 1.1, 1.3, 1.2, 1.0, 1.5, 1.3],
  },
  {
    name: '"photon"',
    blurb: 'strong single peak — one expert dominates',
    logits: [-0.5, 0.2, 3.8, 0.4, 1.1, -0.8, 0.1, 0.5],
  },
  {
    name: '"else:"',
    blurb: 'two clear specialists, rest near zero',
    logits: [-1.0, 3.2, 0.0, -0.3, 0.4, 2.8, 0.1, -0.5],
  },
  {
    name: '"森"',
    blurb: 'bimodal — CJK + multilingual experts both bid',
    logits: [0.1, -0.2, 1.2, 0.3, 0.5, 0.8, 3.1, 2.9],
  },
  {
    name: '"running"',
    blurb: 'three-way tie — router hedges',
    logits: [2.2, 0.3, 0.5, 2.1, -0.4, 2.0, 0.6, 0.4],
  },
]

function softmax(xs: number[]): number[] {
  const m = Math.max(...xs)
  const es = xs.map((x) => Math.exp(x - m))
  const s = es.reduce((a, b) => a + b, 0)
  return es.map((e) => e / s)
}

function topKIdx(xs: number[], k: number): number[] {
  return xs
    .map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v)
    .slice(0, k)
    .map((o) => o.i)
}

export default function RouterGating() {
  const [tokIdx, setTokIdx] = useState(0)
  const [k, setK] = useState(2)

  const tok = TOKENS[tokIdx]
  const { probs, masked, weights, keep } = useMemo(() => {
    const p = softmax(tok.logits)
    const kk = topKIdx(p, k)
    const setKeep = new Set(kk)
    const m = p.map((v, i) => (setKeep.has(i) ? v : 0))
    const sum = m.reduce((a, b) => a + b, 0) || 1
    const w = m.map((v) => v / sum)
    return { probs: p, masked: m, weights: w, keep: setKeep }
  }, [tok, k])

  const maxLogit = Math.max(...tok.logits.map((v) => Math.abs(v)))

  return (
    <WidgetFrame
      widgetName="RouterGating"
      label="router → softmax → top-k → renormalize"
      right={<span className="font-mono">one token, four stages, E = {NUM_EXPERTS}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="k"
            value={k}
            min={1}
            max={4}
            step={1}
            onChange={(v) => setK(Math.round(v))}
            format={(v) => `top-${Math.round(v)}`}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-1.5 flex-wrap">
            {TOKENS.map((t, i) => (
              <button
                key={t.name}
                onClick={() => setTokIdx(i)}
                className={cn(
                  'px-2 py-1 rounded text-[10.5px] font-mono transition-all',
                  tokIdx === i
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {t.name}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="active"
              value={Array.from(keep).sort().map((i) => `E${i}`).join(', ')}
              accent="text-term-amber"
            />
            <Readout
              label="Σ weights"
              value={weights.reduce((a, b) => a + b, 0).toFixed(3)}
              accent="text-term-green"
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-rows-4 gap-3">
        <StageRow
          title="1 · raw logits z_i"
          subtitle={tok.blurb}
          values={tok.logits}
          max={Math.max(4, maxLogit)}
          bipolar
          color="#67e8f9"
          keep={null}
          format={(v) => v.toFixed(2)}
        />
        <StageRow
          title="2 · softmax p_i = exp(z_i) / Σ exp(z_j)"
          subtitle="all positive, sums to 1"
          values={probs}
          max={1}
          color="#fbbf24"
          keep={null}
          format={(v) => v.toFixed(3)}
        />
        <StageRow
          title={`3 · mask — keep top-${k}, rest → 0`}
          subtitle="dropped experts contribute no compute and no weight"
          values={masked}
          max={1}
          color="#f472b6"
          keep={keep}
          format={(v) => (v === 0 ? '·' : v.toFixed(3))}
        />
        <StageRow
          title="4 · renormalize → mixture weights"
          subtitle="weights for the experts that will actually run"
          values={weights}
          max={1}
          color="#4ade80"
          keep={keep}
          format={(v) => (v === 0 ? '·' : v.toFixed(3))}
          highlight
        />
      </div>
    </WidgetFrame>
  )
}

function StageRow({
  title,
  subtitle,
  values,
  max,
  color,
  keep,
  format,
  bipolar,
  highlight,
}: {
  title: string
  subtitle: string
  values: number[]
  max: number
  color: string
  keep: Set<number> | null
  format: (v: number) => string
  bipolar?: boolean
  highlight?: boolean
}) {
  return (
    <div
      className={cn(
        'flex flex-col gap-1 min-h-0',
        highlight && 'bg-dark-surface-elevated/40 rounded-md px-2 py-1 -mx-2',
      )}
    >
      <div className="flex items-center justify-between font-mono">
        <div>
          <div className="text-[11px] text-dark-text-primary">{title}</div>
          <div className="text-[9.5px] text-dark-text-muted">{subtitle}</div>
        </div>
      </div>
      <div className="grid flex-1 gap-1.5 min-h-0" style={{ gridTemplateColumns: `repeat(${NUM_EXPERTS}, 1fr)` }}>
        {values.map((v, i) => {
          const isKept = keep === null || keep.has(i)
          const pct = Math.min(1, Math.abs(v) / max)
          return (
            <div key={i} className="flex flex-col gap-0.5 min-w-0 min-h-0">
              <div className="relative flex-1 bg-dark-surface-elevated/40 rounded-sm overflow-hidden">
                {bipolar && (
                  <div className="absolute left-0 right-0 border-t border-dark-border/60" style={{ top: '50%' }} />
                )}
                <div
                  className="absolute left-0 right-0 transition-all"
                  style={bipolar
                    ? {
                        top: v >= 0 ? `${50 - pct * 50}%` : '50%',
                        height: `${pct * 50}%`,
                        backgroundColor: color,
                        opacity: isKept ? 0.85 : 0.2,
                      }
                    : {
                        bottom: 0,
                        height: `${pct * 100}%`,
                        backgroundColor: color,
                        opacity: isKept ? 0.85 : 0.2,
                      }}
                />
              </div>
              <div className="flex items-center justify-between font-mono text-[9px]">
                <span className={isKept ? 'text-dark-text-secondary' : 'text-dark-text-disabled'}>E{i}</span>
                <span
                  className="tabular-nums"
                  style={{ color: isKept ? color : '#555', opacity: isKept ? 1 : 0.6 }}
                >
                  {format(v)}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
