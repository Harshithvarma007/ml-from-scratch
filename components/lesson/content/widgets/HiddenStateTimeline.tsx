'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Watch a 16-dim hidden state evolve as a char-RNN consumes a string. Each
// column is one time step (one character). Each row is one hidden dimension.
// The colors show signed activation. Click a column to pick a time step — the
// bar chart on the right shows that exact h_t vector.

const H = 16
const SAMPLES = [
  'hello world',
  'the quick fox',
  'rnn loop loop',
]

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gauss(rng: () => number): number {
  const u = Math.max(rng(), 1e-9)
  const v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function charIdx(c: string): number {
  const code = c.toLowerCase().charCodeAt(0)
  if (code >= 97 && code <= 122) return code - 97 // a..z = 0..25
  if (code === 32) return 26                       // space
  return 27                                         // everything else
}

const D = 28

function buildWeights() {
  const rng = mulberry32(42)
  const Wx: number[][] = []
  for (let i = 0; i < H; i++) {
    const row: number[] = []
    for (let j = 0; j < D; j++) row.push(gauss(rng) * 0.5)
    Wx.push(row)
  }
  const Wh: number[][] = []
  for (let i = 0; i < H; i++) {
    const row: number[] = []
    for (let j = 0; j < H; j++) row.push(gauss(rng) * 0.45)
    Wh.push(row)
  }
  const b: number[] = []
  for (let i = 0; i < H; i++) b.push(gauss(rng) * 0.1)
  return { Wx, Wh, b }
}

const { Wx, Wh, b } = buildWeights()

function forward(text: string): number[][] {
  const out: number[][] = []
  let h = new Array(H).fill(0)
  for (const c of text) {
    const idx = charIdx(c)
    const next = new Array(H).fill(0)
    for (let i = 0; i < H; i++) {
      let z = b[i] + Wx[i][idx] // one-hot input → just pick the column
      for (let j = 0; j < H; j++) z += Wh[i][j] * h[j]
      next[i] = Math.tanh(z)
    }
    h = next
    out.push(h)
  }
  return out
}

function activationColor(v: number): string {
  const a = Math.max(0, Math.min(1, Math.abs(v)))
  if (v >= 0) return `rgba(251, 191, 36, ${0.08 + a * 0.85})`
  return `rgba(103, 232, 249, ${0.08 + a * 0.85})`
}

export default function HiddenStateTimeline() {
  const [sampleIdx, setSampleIdx] = useState(0)
  const text = SAMPLES[sampleIdx]
  const H_seq = useMemo(() => forward(text), [text])
  const [sel, setSel] = useState(text.length - 1)
  const clampedSel = Math.min(sel, text.length - 1)
  const h_t = H_seq[clampedSel] ?? new Array(H).fill(0)
  const norm = Math.sqrt(h_t.reduce((a, v) => a + v * v, 0))

  return (
    <WidgetFrame
      widgetName="HiddenStateTimeline"
      label="hidden state as heatmap — dims × time"
      right={<span className="font-mono">char-RNN · H = 16 · tanh</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            {SAMPLES.map((s, i) => (
              <button
                key={s}
                onClick={() => {
                  setSampleIdx(i)
                  setSel(s.length - 1)
                }}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
                  sampleIdx === i
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary',
                )}
              >
                &quot;{s}&quot;
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="t" value={String(clampedSel + 1)} accent="text-term-amber" />
            <Readout label="char" value={`'${text[clampedSel]}'`} />
            <Readout label="‖h_t‖" value={norm.toFixed(3)} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 grid grid-cols-1 md:grid-cols-[1fr_220px] gap-5 overflow-hidden">
        {/* Heatmap */}
        <div className="flex flex-col min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-1">
            h[dim, t] — amber = positive · cyan = negative
          </div>
          <div
            className="grid gap-[2px] p-2 rounded bg-dark-bg flex-1 min-h-0"
            style={{ gridTemplateColumns: `repeat(${text.length}, 1fr)`, gridTemplateRows: `repeat(${H}, 1fr)` }}
          >
            {Array.from({ length: H }).flatMap((_, dim) =>
              text.split('').map((ch, t) => (
                <button
                  key={`${dim}-${t}`}
                  onClick={() => setSel(t)}
                  title={`h[${dim}, ${t}] = ${H_seq[t][dim].toFixed(3)}`}
                  className={cn(
                    'rounded-[2px]',
                    clampedSel === t ? 'outline outline-1 outline-term-amber' : '',
                  )}
                  style={{ backgroundColor: activationColor(H_seq[t][dim]) }}
                />
              )),
            )}
          </div>
          <div
            className="grid px-2 pt-1.5 font-mono text-[10px] text-dark-text-muted"
            style={{ gridTemplateColumns: `repeat(${text.length}, 1fr)` }}
          >
            {text.split('').map((ch, t) => (
              <div
                key={t}
                onClick={() => setSel(t)}
                className={cn(
                  'text-center cursor-pointer',
                  clampedSel === t ? 'text-term-amber' : 'hover:text-dark-text-primary',
                )}
              >
                {ch === ' ' ? '·' : ch}
              </div>
            ))}
          </div>
        </div>

        {/* Right: vector preview */}
        <div className="flex flex-col gap-2 min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            h_{clampedSel + 1} = h[:, {clampedSel}]
          </div>
          <div className="flex flex-col gap-[2px] flex-1 justify-center">
            {h_t.map((v, i) => (
              <div key={i} className="flex items-center gap-1.5 font-mono text-[9.5px] text-dark-text-muted">
                <span className="w-5 text-right">{i}</span>
                <div className="flex-1 h-3 bg-dark-surface-elevated/40 rounded-sm relative overflow-hidden">
                  <div
                    className="absolute top-0 bottom-0"
                    style={{
                      left: v >= 0 ? '50%' : `${50 + v * 50}%`,
                      width: `${Math.abs(v) * 50}%`,
                      backgroundColor: v >= 0 ? 'rgba(251, 191, 36, 0.75)' : 'rgba(103, 232, 249, 0.75)',
                    }}
                  />
                  <div className="absolute top-0 bottom-0 border-l border-dark-border" style={{ left: '50%' }} />
                </div>
                <span className="w-11 text-right tabular-nums">{v.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
