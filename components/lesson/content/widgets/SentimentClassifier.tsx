'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Logistic-regression-style sentiment classifier with hand-picked word weights.
// Shows each word's contribution as a colored chip (green = positive logit,
// rose = negative logit), accumulates the logit sum, and reports σ(sum) as
// the predicted P(positive). Pre-loaded with four sample reviews.

const WEIGHTS: Record<string, number> = {
  // positives
  love: 1.8, great: 1.5, amazing: 1.7, excellent: 1.9, wonderful: 1.6,
  brilliant: 1.4, perfect: 1.7, fantastic: 1.6, enjoyed: 1.2, good: 0.9,
  // negatives
  hate: -1.9, awful: -1.8, terrible: -1.9, worst: -2.0, disappointing: -1.6,
  boring: -1.4, bad: -1.0, horrible: -1.8, dull: -1.1, waste: -1.5,
}

const SAMPLES: readonly string[] = [
  'I absolutely love this movie, it was amazing and wonderful',
  'A dull and boring film, honestly a waste of time',
  'The acting was good but the plot was terrible and disappointing',
  'Fantastic performances, brilliant direction, simply perfect',
]

const BIAS = -0.1

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z))
}

type WordHit = { token: string; weight: number }

function scoreText(text: string): { hits: WordHit[]; logit: number } {
  const words = text.toLowerCase().match(/[a-z]+/g) ?? []
  const hits: WordHit[] = []
  let logit = BIAS
  for (const w of words) {
    const weight = WEIGHTS[w]
    hits.push({ token: w, weight: weight ?? 0 })
    if (weight !== undefined) logit += weight
  }
  return { hits, logit }
}

export default function SentimentClassifier() {
  const [text, setText] = useState<string>(SAMPLES[0])

  const { hits, logit } = useMemo(() => scoreText(text), [text])
  const p = sigmoid(logit)
  const posW = hits.filter((h) => h.weight > 0).length
  const negW = hits.filter((h) => h.weight < 0).length

  return (
    <WidgetFrame
      widgetName="SentimentClassifier"
      label="sentiment classifier — logit = bias + Σ w_i · x_i"
      right={<span className="font-mono">log-linear · 20 hand-weighted words · σ(logit) = P(positive)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">reviews:</span>
          {SAMPLES.map((s, i) => (
            <button
              key={i}
              onClick={() => setText(s)}
              className={cn(
                'px-2 py-1 rounded text-[10px] font-mono transition-all',
                text === s
                  ? 'bg-dark-accent text-white'
                  : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
              )}
            >
              {i + 1}
            </button>
          ))}
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="logit" value={logit.toFixed(2)} accent={logit >= 0 ? 'text-term-green' : 'text-term-rose'} />
            <Readout
              label="P(pos)"
              value={`${(p * 100).toFixed(1)}%`}
              accent={p > 0.5 ? 'text-term-green' : 'text-term-rose'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 flex flex-col gap-3 overflow-auto">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          spellCheck={false}
          rows={2}
          className="w-full bg-dark-surface-elevated border border-dark-border rounded px-3 py-2 font-mono text-[12.5px] text-dark-text-primary outline-none focus:border-dark-border-hover resize-none"
        />

        <div className="grid grid-cols-1 md:grid-cols-[1fr_260px] gap-4 flex-1 min-h-0">
          {/* Token chips with weights */}
          <div className="bg-dark-surface-elevated/30 border border-dark-border rounded p-3 overflow-auto">
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
              token weights (grey = not in lexicon, neutral contribution)
            </div>
            <div className="flex flex-wrap gap-1.5">
              {hits.map((h, i) => {
                const pos = h.weight > 0
                const neg = h.weight < 0
                const zero = h.weight === 0
                const mag = Math.min(1, Math.abs(h.weight) / 2)
                const bg = zero
                  ? 'rgba(100,100,100,0.12)'
                  : pos
                  ? `rgba(74, 222, 128, ${0.15 + mag * 0.5})`
                  : `rgba(248, 113, 113, ${0.15 + mag * 0.5})`
                const border = zero ? '#333' : pos ? 'rgba(74,222,128,0.7)' : 'rgba(248,113,113,0.7)'
                const col = zero ? '#888' : pos ? '#4ade80' : '#f87171'
                return (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1.5 rounded px-2 py-1 font-mono text-[11px]"
                    style={{ backgroundColor: bg, border: `1px solid ${border}`, color: col }}
                  >
                    <span>{h.token}</span>
                    {!zero && (
                      <span className="text-[9.5px] tabular-nums text-dark-text-disabled">
                        {h.weight >= 0 ? '+' : ''}
                        {h.weight.toFixed(1)}
                      </span>
                    )}
                  </span>
                )
              })}
            </div>
          </div>

          {/* Logit breakdown + probability */}
          <div className="flex flex-col gap-2">
            <div className="bg-dark-surface-elevated/40 border border-dark-border rounded p-3 flex flex-col gap-1.5">
              <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
                logit breakdown
              </div>
              <div className="flex justify-between font-mono text-[11px]">
                <span className="text-dark-text-muted">bias</span>
                <span className="tabular-nums">{BIAS.toFixed(2)}</span>
              </div>
              <div className="flex justify-between font-mono text-[11px]">
                <span className="text-term-green">+ positives ({posW})</span>
                <span className="tabular-nums text-term-green">
                  +{hits.filter((h) => h.weight > 0).reduce((a, h) => a + h.weight, 0).toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between font-mono text-[11px]">
                <span className="text-term-rose">− negatives ({negW})</span>
                <span className="tabular-nums text-term-rose">
                  {hits.filter((h) => h.weight < 0).reduce((a, h) => a + h.weight, 0).toFixed(2)}
                </span>
              </div>
              <div className="border-t border-dark-border my-0.5" />
              <div className="flex justify-between font-mono text-[11.5px]">
                <span className="text-dark-text-primary">Σ logit</span>
                <span className={cn('tabular-nums', logit >= 0 ? 'text-term-green' : 'text-term-rose')}>
                  {logit.toFixed(2)}
                </span>
              </div>
            </div>

            {/* Probability bar */}
            <div className="bg-dark-surface-elevated/40 border border-dark-border rounded p-3 flex flex-col gap-2">
              <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
                σ(logit) → P(positive)
              </div>
              <div className="relative h-6 bg-dark-bg rounded overflow-hidden border border-dark-border">
                <div
                  className="absolute top-0 bottom-0 left-0"
                  style={{
                    width: `${p * 100}%`,
                    background: 'linear-gradient(90deg, rgba(74,222,128,0.35), rgba(74,222,128,0.9))',
                  }}
                />
                <div className="absolute inset-0 flex items-center justify-center font-mono text-[11px] text-white">
                  {(p * 100).toFixed(1)}% positive · {((1 - p) * 100).toFixed(1)}% negative
                </div>
              </div>
              <div className="text-[10.5px] font-mono text-dark-text-muted">
                verdict:{' '}
                <span className={p > 0.5 ? 'text-term-green' : 'text-term-rose'}>
                  {p > 0.5 ? 'positive' : 'negative'}
                </span>
                {Math.abs(p - 0.5) < 0.1 && <span className="text-term-amber"> (hedging)</span>}
              </div>
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
