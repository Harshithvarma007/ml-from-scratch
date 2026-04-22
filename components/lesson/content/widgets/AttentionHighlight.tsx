'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Same lexicon as the classifier, but visualized as attention. Each word gets
// a softmaxed weight derived from |w_i| — the model "looks" harder at the
// sentimental words. Toggle between class-colored mode (green/rose) and
// pure-intensity attention mode.

const WEIGHTS: Record<string, number> = {
  love: 1.8, great: 1.5, amazing: 1.7, excellent: 1.9, wonderful: 1.6,
  brilliant: 1.4, perfect: 1.7, fantastic: 1.6, enjoyed: 1.2, good: 0.9,
  hate: -1.9, awful: -1.8, terrible: -1.9, worst: -2.0, disappointing: -1.6,
  boring: -1.4, bad: -1.0, horrible: -1.8, dull: -1.1, waste: -1.5,
}

const SAMPLES: readonly string[] = [
  'the movie was absolutely wonderful and i love it',
  'a truly terrible and boring film, the worst i have seen',
  'good acting cannot save a disappointing and dull script',
  'fantastic direction, brilliant writing, perfect pacing',
]

type Mode = 'class' | 'attention'

function softmax(x: number[]): number[] {
  const maxX = Math.max(...x)
  const exps = x.map((v) => Math.exp(v - maxX))
  const sum = exps.reduce((a, v) => a + v, 0)
  return exps.map((v) => v / sum)
}

type Token = { word: string; weight: number; attn: number }

function analyze(text: string, temperature: number): { tokens: Token[]; maxAttn: number } {
  const words = text.match(/[A-Za-z]+|[^\sA-Za-z]/g) ?? []
  const raw = words.map((w) => Math.abs(WEIGHTS[w.toLowerCase()] ?? 0) / temperature)
  const probs = softmax(raw.length === 0 ? [0] : raw)
  const tokens: Token[] = words.map((w, i) => ({
    word: w,
    weight: WEIGHTS[w.toLowerCase()] ?? 0,
    attn: probs[i] ?? 0,
  }))
  return { tokens, maxAttn: Math.max(...tokens.map((t) => t.attn), 1e-6) }
}

export default function AttentionHighlight() {
  const [idx, setIdx] = useState(0)
  const [mode, setMode] = useState<Mode>('attention')
  const [temperature, setTemperature] = useState(1.0)

  const text = SAMPLES[idx]
  const { tokens, maxAttn } = useMemo(() => analyze(text, temperature), [text, temperature])
  const top = [...tokens].sort((a, b) => b.attn - a.attn).slice(0, 3)

  return (
    <WidgetFrame
      widgetName="AttentionHighlight"
      label="attention overlay — whose word weights count?"
      right={<span className="font-mono">softmax(|w_i| / τ) over the sentence · τ controls focus sharpness</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5">
            {SAMPLES.map((_, i) => (
              <button
                key={i}
                onClick={() => setIdx(i)}
                className={cn(
                  'px-2 py-1 rounded text-[10px] font-mono transition-all',
                  idx === i
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                sent {i + 1}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1">
            {(['attention', 'class'] as Mode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={cn(
                  'px-2 py-1 rounded text-[10px] font-mono uppercase transition-all',
                  mode === m
                    ? 'bg-term-amber/20 text-term-amber border border-term-amber/60'
                    : 'border border-dark-border text-dark-text-secondary',
                )}
              >
                {m}
              </button>
            ))}
          </div>
          <label className="flex items-center gap-2 font-mono text-[11px]">
            <span className="text-dark-text-secondary">τ</span>
            <input
              type="range"
              min={0.3}
              max={3}
              step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="h-1 w-28 rounded-full bg-dark-border cursor-pointer accent-dark-accent"
            />
            <span className="tabular-nums w-8 text-right">{temperature.toFixed(2)}</span>
          </label>
          <div className="ml-auto">
            <Readout label="peak" value={`${(maxAttn * 100).toFixed(1)}%`} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-rows-[1fr_auto] gap-3 overflow-hidden">
        {/* Sentence view */}
        <div className="bg-dark-surface-elevated/30 border border-dark-border rounded p-4 overflow-auto flex flex-col justify-center">
          <div className="flex flex-wrap gap-x-1 gap-y-3 leading-relaxed">
            {tokens.map((t, i) => {
              const inten = t.attn / maxAttn
              const bg = mode === 'attention'
                ? `rgba(251, 191, 36, ${0.08 + inten * 0.65})`
                : t.weight > 0
                ? `rgba(74, 222, 128, ${0.1 + inten * 0.6})`
                : t.weight < 0
                ? `rgba(248, 113, 113, ${0.1 + inten * 0.6})`
                : 'rgba(100,100,100,0.08)'
              const under = mode === 'attention'
                ? '#fbbf24'
                : t.weight > 0 ? '#4ade80' : t.weight < 0 ? '#f87171' : '#555'
              return (
                <span key={i} className="inline-flex flex-col items-stretch" style={{ minWidth: 24 }}>
                  <span
                    className="font-mono text-[14.5px] px-1.5 py-0.5 rounded text-dark-text-primary"
                    style={{ backgroundColor: bg }}
                  >
                    {t.word}
                  </span>
                  <span
                    className="h-1 mt-0.5 rounded-full"
                    style={{ width: `${inten * 100}%`, backgroundColor: under, opacity: 0.85 }}
                  />
                  <span className="text-[9px] font-mono text-dark-text-disabled tabular-nums text-center mt-0.5">
                    {(t.attn * 100).toFixed(0)}%
                  </span>
                </span>
              )
            })}
          </div>
        </div>

        {/* Top-3 attention readout */}
        <div className="flex items-center gap-3 font-mono text-[11px]">
          <span className="text-[10px] uppercase tracking-wider text-dark-text-disabled">top attention →</span>
          {top.map((t, i) => {
            const col = mode === 'attention'
              ? 'text-term-amber'
              : t.weight > 0 ? 'text-term-green' : t.weight < 0 ? 'text-term-rose' : 'text-dark-text-muted'
            return (
              <span key={i} className="flex items-center gap-1.5">
                <span className="text-dark-text-disabled">#{i + 1}</span>
                <span className={col}>{t.word}</span>
                <span className="tabular-nums text-dark-text-muted">
                  ({(t.attn * 100).toFixed(1)}%)
                </span>
              </span>
            )
          })}
        </div>
      </div>
    </WidgetFrame>
  )
}
