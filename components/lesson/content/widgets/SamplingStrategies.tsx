'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Four sampling strategies on the same pre-softmax logits. Four small panels,
// each shows the strategy's resulting distribution. Tokens kept are colored;
// zeroed tokens are faded. Readouts for effective vocab size per strategy.

const TOKENS = ['the', ' a', ' cat', ' dog', ' sat', ' ran', ' quick', ' slow', '!', '.']
const LOGITS = [3.4, 2.8, 2.1, 1.6, 1.0, 0.4, -0.2, -0.8, -1.3, -2.0]

type Strategy = 'greedy' | 'temp' | 'topk' | 'nucleus'

function softmax(x: number[]): number[] {
  const m = Math.max(...x)
  const e = x.map((v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

function entropy(p: number[]): number {
  return -p.reduce((a, v) => a + (v > 1e-12 ? v * Math.log(v) : 0), 0)
}

function effectiveVocab(p: number[]): number {
  return p.reduce((a, v) => a + (v > 1e-6 ? 1 : 0), 0)
}

function greedy(logits: number[]): { p: number[]; kept: boolean[] } {
  const argmax = logits.indexOf(Math.max(...logits))
  const p = logits.map((_, i) => (i === argmax ? 1 : 0))
  const kept = logits.map((_, i) => i === argmax)
  return { p, kept }
}

function temperature(logits: number[], T: number): { p: number[]; kept: boolean[] } {
  const scaled = logits.map((v) => v / Math.max(T, 1e-6))
  return { p: softmax(scaled), kept: logits.map(() => true) }
}

function topK(logits: number[], k: number): { p: number[]; kept: boolean[] } {
  const sorted = logits.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v)
  const top = new Set(sorted.slice(0, k).map((x) => x.i))
  const masked = logits.map((v, i) => (top.has(i) ? v : -Infinity))
  return { p: softmax(masked), kept: logits.map((_, i) => top.has(i)) }
}

function nucleus(logits: number[], p: number): { p: number[]; kept: boolean[] } {
  const probs = softmax(logits)
  const sorted = probs.map((pr, i) => ({ pr, i })).sort((a, b) => b.pr - a.pr)
  let cum = 0
  const kept = new Set<number>()
  for (const { pr, i } of sorted) {
    kept.add(i)
    cum += pr
    if (cum >= p) break
  }
  const masked = logits.map((v, i) => (kept.has(i) ? v : -Infinity))
  return { p: softmax(masked), kept: logits.map((_, i) => kept.has(i)) }
}

export default function SamplingStrategies() {
  const [T, setT] = useState(1.0)
  const [K, setK] = useState(5)
  const [P, setP] = useState(0.9)
  const [active, setActive] = useState<Strategy>('nucleus')

  const origP = useMemo(() => softmax(LOGITS), [])

  const greedyOut = useMemo(() => greedy(LOGITS), [])
  const tempOut = useMemo(() => temperature(LOGITS, T), [T])
  const topkOut = useMemo(() => topK(LOGITS, K), [K])
  const nucOut = useMemo(() => nucleus(LOGITS, P), [P])

  const strategies: { key: Strategy; name: string; color: string; out: { p: number[]; kept: boolean[] }; sub: string }[] = [
    { key: 'greedy', name: 'greedy', color: '#f87171', out: greedyOut, sub: 'argmax — zero randomness' },
    { key: 'temp', name: `T = ${T.toFixed(2)}`, color: '#fbbf24', out: tempOut, sub: 'softmax(logits / T)' },
    { key: 'topk', name: `top-k = ${K}`, color: '#67e8f9', out: topkOut, sub: `keep top ${K}, re-normalize` },
    { key: 'nucleus', name: `top-p = ${P.toFixed(2)}`, color: '#a78bfa', out: nucOut, sub: `smallest set with p ≥ ${P.toFixed(2)}` },
  ]

  return (
    <WidgetFrame
      widgetName="SamplingStrategies"
      label="four samplers on the same logits"
      right={<span className="font-mono">vocab = {TOKENS.length} · underlying logits fixed</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider label="T" value={T} min={0.1} max={2.0} step={0.05} onChange={setT} format={(v) => v.toFixed(2)} accent="accent-term-amber" />
          <Slider label="k" value={K} min={1} max={10} step={1} onChange={(v) => setK(Math.round(v))} format={(v) => String(Math.round(v))} accent="accent-term-cyan" />
          <Slider label="p" value={P} min={0.1} max={1.0} step={0.05} onChange={setP} format={(v) => v.toFixed(2)} accent="accent-term-purple" />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="H(orig)" value={entropy(origP).toFixed(2)} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[220px_1fr] gap-4 overflow-hidden">
        {/* Left: original logits */}
        <div className="flex flex-col gap-2 min-w-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            pre-softmax logits
          </div>
          <div className="flex flex-col gap-1 font-mono text-[10px] bg-dark-surface-elevated/40 rounded p-2">
            {TOKENS.map((t, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-12 text-dark-text-secondary truncate">&quot;{t}&quot;</span>
                <div className="flex-1 h-2 bg-dark-bg rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-term-amber/70"
                    style={{ width: `${((LOGITS[i] - Math.min(...LOGITS)) / (Math.max(...LOGITS) - Math.min(...LOGITS))) * 100}%` }}
                  />
                </div>
                <span className="w-10 text-right tabular-nums text-dark-text-primary">{LOGITS[i].toFixed(1)}</span>
              </div>
            ))}
          </div>
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-2">
            original probabilities (T=1)
          </div>
          <div className="flex flex-col gap-0.5 font-mono text-[9.5px]">
            {TOKENS.map((t, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-10 text-dark-text-muted truncate">&quot;{t}&quot;</span>
                <div className="flex-1 h-1.5 bg-dark-surface-elevated/40 rounded-full overflow-hidden">
                  <div className="h-full rounded-full bg-term-green/70" style={{ width: `${origP[i] * 100}%` }} />
                </div>
                <span className="w-9 text-right tabular-nums text-dark-text-muted">{(origP[i] * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Right: 4 strategy panels */}
        <div className="grid grid-cols-2 gap-3 min-h-0 overflow-auto">
          {strategies.map((s) => (
            <StrategyPanel key={s.key} s={s} active={active === s.key} onSelect={() => setActive(s.key)} />
          ))}
        </div>
      </div>
    </WidgetFrame>
  )
}

function StrategyPanel({
  s,
  active,
  onSelect,
}: {
  s: {
    key: Strategy
    name: string
    color: string
    sub: string
    out: { p: number[]; kept: boolean[] }
  }
  active: boolean
  onSelect: () => void
}) {
  const H = entropy(s.out.p)
  const vocab = effectiveVocab(s.out.p)
  return (
    <button
      onClick={onSelect}
      className={cn(
        'text-left rounded border p-2.5 flex flex-col gap-1.5 min-w-0 transition-all',
        active ? 'bg-dark-surface-elevated/60' : 'bg-dark-surface-elevated/20 hover:bg-dark-surface-elevated/40',
      )}
      style={{ borderColor: active ? s.color : 'rgba(63,63,70,0.7)' }}
    >
      <div className="flex items-center justify-between font-mono text-[10.5px]">
        <span style={{ color: s.color }}>{s.name}</span>
        <span className="text-dark-text-disabled text-[9.5px]">
          H = {H.toFixed(2)} · eff = {vocab}
        </span>
      </div>
      <div className="text-[9.5px] font-mono text-dark-text-disabled">{s.sub}</div>
      <div className="flex flex-col gap-0.5 mt-1">
        {TOKENS.map((t, i) => {
          const kept = s.out.kept[i]
          return (
            <div key={i} className="flex items-center gap-1.5 font-mono text-[9px]">
              <span className={cn('w-8 truncate', kept ? 'text-dark-text-secondary' : 'text-dark-text-disabled')}>
                &quot;{t}&quot;
              </span>
              <div className="flex-1 h-1.5 bg-dark-bg rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${s.out.p[i] * 100}%`,
                    backgroundColor: kept ? s.color : '#3f3f46',
                    opacity: kept ? 0.85 : 0.3,
                  }}
                />
              </div>
              <span className={cn('w-8 text-right tabular-nums', kept ? 'text-dark-text-primary' : 'text-dark-text-disabled')}>
                {(s.out.p[i] * 100).toFixed(0)}
              </span>
            </div>
          )
        })}
      </div>
    </button>
  )
}
