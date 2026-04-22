'use client'

// Step through BPE training on "the quick brown fox jumps". Each click finds
// the most frequent adjacent pair in the current sequence, merges every
// occurrence into a new token, and appends that token to a growing vocab.
// Left: pipeline (before → merged pair highlighted → after). Right: vocab
// list scrolls as tokens get added.

import { useMemo, useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { SkipBack, StepBack, StepForward } from 'lucide-react'
import { cn } from '@/lib/utils'

const CORPUS = 'the quick brown fox jumps'

type MergeStep = {
  // tokens after this step's merge has been applied
  tokens: string[]
  // the pair we merged this step (e.g., ['t','h'] -> 'th')
  pair: [string, string] | null
  merged: string | null
  // index positions in the PREVIOUS token list that were merged
  mergedAt: number[]
  // all new tokens added to vocab by end of this step
  vocab: string[]
  // frequency of the most-frequent pair we chose
  freq: number
}

function countPairs(tokens: string[]): Map<string, number> {
  const counts = new Map<string, number>()
  for (let i = 0; i < tokens.length - 1; i++) {
    const a = tokens[i]
    const b = tokens[i + 1]
    // don't merge across the space boundary: pretend space is an "end" marker
    if (a === ' ' || b === ' ') continue
    const k = `${a}\u0001${b}`
    counts.set(k, (counts.get(k) ?? 0) + 1)
  }
  return counts
}

function pickBestPair(tokens: string[]): { pair: [string, string]; freq: number } | null {
  const counts = countPairs(tokens)
  let best: [string, string] | null = null
  let bestFreq = 0
  for (const [k, v] of counts.entries()) {
    if (v > bestFreq) {
      const [a, b] = k.split('\u0001')
      best = [a, b]
      bestFreq = v
    }
  }
  if (!best) return null
  return { pair: best, freq: bestFreq }
}

function applyMerge(tokens: string[], pair: [string, string]): { next: string[]; positions: number[] } {
  const [a, b] = pair
  const out: string[] = []
  const positions: number[] = []
  let i = 0
  while (i < tokens.length) {
    if (i < tokens.length - 1 && tokens[i] === a && tokens[i + 1] === b) {
      positions.push(i)
      out.push(a + b)
      i += 2
    } else {
      out.push(tokens[i])
      i += 1
    }
  }
  return { next: out, positions }
}

function buildMergeSteps(): { initial: string[]; steps: MergeStep[]; initialVocab: string[] } {
  const initial = Array.from(CORPUS)
  const baseVocab: string[] = []
  for (const c of initial) if (!baseVocab.includes(c)) baseVocab.push(c)
  const steps: MergeStep[] = []
  let tokens = initial.slice()
  let vocab = baseVocab.slice()
  for (let s = 0; s < 10; s++) {
    const best = pickBestPair(tokens)
    if (!best || best.freq < 2) break
    const { next, positions } = applyMerge(tokens, best.pair)
    const merged = best.pair[0] + best.pair[1]
    vocab = vocab.includes(merged) ? vocab : [...vocab, merged]
    steps.push({
      tokens: next,
      pair: best.pair,
      merged,
      mergedAt: positions,
      vocab: vocab.slice(),
      freq: best.freq,
    })
    tokens = next
  }
  return { initial, steps, initialVocab: baseVocab }
}

const COLORS = ['#fbbf24', '#67e8f9', '#a78bfa', '#f472b6', '#4ade80', '#f87171', '#5eead4']

function tokenColor(tok: string, vocab: string[]): string {
  // single chars stay muted; merged tokens get accent colors
  if (tok.length <= 1) return '#444'
  const idx = vocab.indexOf(tok)
  const mergedIdx = Math.max(0, idx - 27)
  return COLORS[mergedIdx % COLORS.length]
}

export default function BPEMergesAnimation() {
  const { initial, steps, initialVocab } = useMemo(() => buildMergeSteps(), [])
  const [i, setI] = useState(0)

  // i = 0 means "no merges yet", i = 1..steps.length means "applied merges[0..i-1]"
  const prevTokens = i === 0 ? initial : i === 1 ? initial : steps[i - 2].tokens
  const currTokens = i === 0 ? initial : steps[i - 1].tokens
  const currStep = i === 0 ? null : steps[i - 1]
  const vocab = i === 0 ? initialVocab : steps[i - 1].vocab

  const pairStr = currStep?.pair ? `"${currStep.pair[0]}" + "${currStep.pair[1]}"` : '—'

  return (
    <WidgetFrame
      widgetName="BPEMergesAnimation"
      label='byte-pair encoding on "the quick brown fox jumps"'
      right={<span className="font-mono">base vocab 27 chars · step up to 10 merges</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <Button onClick={() => setI(0)} variant="ghost">
              <span className="inline-flex items-center gap-1">
                <SkipBack size={11} /> reset
              </span>
            </Button>
            <Button onClick={() => setI(Math.max(0, i - 1))} variant="ghost" disabled={i === 0}>
              <span className="inline-flex items-center gap-1">
                <StepBack size={11} /> back
              </span>
            </Button>
            <Button
              onClick={() => setI(Math.min(steps.length, i + 1))}
              variant="primary"
              disabled={i === steps.length}
            >
              <span className="inline-flex items-center gap-1">
                merge <StepForward size={11} />
              </span>
            </Button>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="step" value={`${i} / ${steps.length}`} />
            <Readout label="pair" value={pairStr} accent="text-term-amber" />
            <Readout label="freq" value={String(currStep?.freq ?? 0)} accent="text-term-green" />
            <Readout label="|V|" value={String(vocab.length)} accent="text-term-cyan" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[1fr_220px] gap-4 overflow-hidden">
        {/* Left: pipeline */}
        <div className="flex flex-col gap-3 min-h-0 overflow-hidden">
          <PipelineRow
            title="before"
            tokens={prevTokens}
            highlightAt={currStep?.mergedAt ?? []}
            highlightLen={2}
            vocab={vocab}
          />
          <div className="flex items-center justify-center text-[11px] font-mono text-term-amber">
            {currStep?.pair
              ? `merge "${currStep.pair[0]}" + "${currStep.pair[1]}" → "${currStep.merged}"  (×${currStep.freq})`
              : '↓  pick the highest-frequency adjacent pair  ↓'}
          </div>
          <PipelineRow
            title="after"
            tokens={currTokens}
            highlightAt={
              currStep
                ? currStep.mergedAt.map((p, k) => p - k) // after merge, indices collapse
                : []
            }
            highlightLen={1}
            highlightMerged
            vocab={vocab}
          />
          <div className="mt-2 text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            counts (top adjacent pairs in the &quot;after&quot; row)
          </div>
          <TopPairs tokens={currTokens} />
        </div>

        {/* Right: vocab list */}
        <div className="flex flex-col gap-2 min-h-0 overflow-hidden">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            vocabulary · {vocab.length} tokens
          </div>
          <div className="flex-1 overflow-auto border border-dark-border rounded bg-dark-surface-elevated/20 p-2 font-mono text-[10.5px]">
            <div className="grid grid-cols-1 md:grid-cols-[32px_1fr] gap-y-[3px] gap-x-2">
              {vocab.map((tok, idx) => {
                const isNew =
                  currStep &&
                  idx === vocab.length - 1 &&
                  tok === currStep.merged
                return (
                  <div key={idx} className="contents">
                    <span className="text-dark-text-disabled tabular-nums text-right">{idx}</span>
                    <span
                      className={cn('truncate', isNew && 'text-term-amber')}
                      style={{ color: isNew ? undefined : tokenColor(tok, vocab) }}
                    >
                      {tok === ' ' ? '·space·' : `"${tok}"`}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function PipelineRow({
  title,
  tokens,
  highlightAt,
  highlightLen,
  highlightMerged,
  vocab,
}: {
  title: string
  tokens: string[]
  highlightAt: number[]
  highlightLen: number
  highlightMerged?: boolean
  vocab: string[]
}) {
  const highlightSet = new Set<number>()
  for (const p of highlightAt) {
    for (let k = 0; k < highlightLen; k++) highlightSet.add(p + k)
  }
  return (
    <div className="flex flex-col gap-1.5">
      <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
        {title} · {tokens.length} tokens
      </div>
      <div className="flex flex-wrap gap-[3px] p-2 bg-dark-surface-elevated/30 border border-dark-border rounded">
        {tokens.map((tok, idx) => {
          const active = highlightSet.has(idx)
          const color = tokenColor(tok, vocab)
          return (
            <span
              key={idx}
              className={cn(
                'font-mono text-[12px] px-1.5 py-[2px] rounded-sm border transition-colors',
                active
                  ? highlightMerged
                    ? 'border-term-amber bg-term-amber/20 text-term-amber'
                    : 'border-term-rose bg-term-rose/15 text-term-rose'
                  : 'border-dark-border/60 bg-dark-bg',
              )}
              style={!active ? { color } : undefined}
            >
              {tok === ' ' ? '\u00b7' : tok}
            </span>
          )
        })}
      </div>
    </div>
  )
}

function TopPairs({ tokens }: { tokens: string[] }) {
  const counts = countPairs(tokens)
  const entries = Array.from(counts.entries())
    .map(([k, v]) => {
      const [a, b] = k.split('\u0001')
      return { a, b, v }
    })
    .sort((x, y) => y.v - x.v)
    .slice(0, 6)
  const max = entries[0]?.v ?? 1
  return (
    <div className="flex flex-col gap-1">
      {entries.length === 0 && (
        <div className="text-[11px] font-mono text-dark-text-disabled">no adjacent pairs remain</div>
      )}
      {entries.map(({ a, b, v }, idx) => (
        <div key={idx} className="flex items-center gap-2 font-mono text-[10.5px]">
          <span className="w-28 text-dark-text-secondary truncate">
            &quot;{a}&quot; + &quot;{b}&quot;
          </span>
          <div className="flex-1 h-2 bg-dark-surface-elevated/40 rounded-sm overflow-hidden">
            <div
              className="h-full"
              style={{
                width: `${(v / max) * 100}%`,
                backgroundColor: idx === 0 ? '#fbbf24' : '#555',
                opacity: idx === 0 ? 0.85 : 0.55,
              }}
            />
          </div>
          <span className="w-6 text-right tabular-nums text-dark-text-muted">{v}</span>
        </div>
      ))}
    </div>
  )
}
