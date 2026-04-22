'use client'

import { Fragment, useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// 4 hand-crafted attention heatmaps showing that different heads learn
// different jobs: one tracks the previous token, one aligns same-POS words,
// one points back to the subject, one is a positional/broadcast head. Toggle
// through 3 sentences to see each head's pattern adapt.

type POS = 'DET' | 'NOUN' | 'VERB' | 'PREP' | 'ADJ' | 'ADV'

type Sentence = {
  label: string
  tokens: string[]
  pos: POS[]
  subject: number // index of the sentence's main subject
}

const SENTENCES: Sentence[] = [
  {
    label: 'the cat sat on the mat yesterday afternoon',
    tokens: ['the', 'cat', 'sat', 'on', 'the', 'mat', 'yesterday', 'afternoon'],
    pos: ['DET', 'NOUN', 'VERB', 'PREP', 'DET', 'NOUN', 'ADV', 'NOUN'],
    subject: 1,
  },
  {
    label: 'she quickly ran through the dark forest alone',
    tokens: ['she', 'quickly', 'ran', 'through', 'the', 'dark', 'forest', 'alone'],
    pos: ['NOUN', 'ADV', 'VERB', 'PREP', 'DET', 'ADJ', 'NOUN', 'ADV'],
    subject: 0,
  },
  {
    label: 'the bright yellow sun warmed the open field',
    tokens: ['the', 'bright', 'yellow', 'sun', 'warmed', 'the', 'open', 'field'],
    pos: ['DET', 'ADJ', 'ADJ', 'NOUN', 'VERB', 'DET', 'ADJ', 'NOUN'],
    subject: 3,
  },
]

type HeadSpec = {
  name: string
  hue: string
  blurb: string
  build: (s: Sentence) => number[][]
}

function softmaxRow(row: number[]): number[] {
  const m = Math.max(...row)
  const e = row.map((v) => Math.exp(v - m))
  const sum = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / sum)
}

function causalSoftmax(scores: number[][]): number[][] {
  return scores.map((row, i) =>
    softmaxRow(row.map((v, j) => (j > i ? -1e9 : v))),
  )
}

const HEADS: HeadSpec[] = [
  {
    name: 'previous-token',
    hue: '#67e8f9',
    blurb: 'peaks at j = i − 1. The bigram workhorse.',
    build: (s) => {
      const N = s.tokens.length
      const S: number[][] = []
      for (let i = 0; i < N; i++) {
        const row: number[] = []
        for (let j = 0; j < N; j++) {
          row.push(j === i - 1 ? 5 : j === i ? 0.5 : -1.5 - Math.abs(i - j - 1) * 0.5)
        }
        S.push(row)
      }
      return causalSoftmax(S)
    },
  },
  {
    name: 'same-POS',
    hue: '#a78bfa',
    blurb: 'boosts earlier tokens with matching part-of-speech tag.',
    build: (s) => {
      const N = s.tokens.length
      const S: number[][] = []
      for (let i = 0; i < N; i++) {
        const row: number[] = []
        for (let j = 0; j < N; j++) {
          row.push(s.pos[i] === s.pos[j] ? 3 : -0.5)
        }
        S.push(row)
      }
      return causalSoftmax(S)
    },
  },
  {
    name: 'subject-pointer',
    hue: '#4ade80',
    blurb: 'every verb/object points back to the sentence subject.',
    build: (s) => {
      const N = s.tokens.length
      const S: number[][] = []
      for (let i = 0; i < N; i++) {
        const row: number[] = []
        for (let j = 0; j < N; j++) {
          const isSubj = j === s.subject
          const selfBias = i === j ? 0.8 : 0
          row.push(isSubj && i > s.subject ? 3.5 : selfBias - Math.abs(i - j) * 0.2)
        }
        S.push(row)
      }
      return causalSoftmax(S)
    },
  },
  {
    name: 'positional-bias',
    hue: '#fbbf24',
    blurb: 'diffuse, leans on sentence start — a broadcast/summary head.',
    build: (s) => {
      const N = s.tokens.length
      const S: number[][] = []
      for (let i = 0; i < N; i++) {
        const row: number[] = []
        for (let j = 0; j < N; j++) {
          // pull toward j = 0 lightly, plus a slight self-smoothing
          row.push(-0.1 * j + (i === j ? 0.5 : 0))
        }
        S.push(row)
      }
      return causalSoftmax(S)
    },
  },
]

export default function HeadSpecialization() {
  const [sIdx, setSIdx] = useState(0)
  const sentence = SENTENCES[sIdx]
  const heads = useMemo(() => HEADS.map((h) => h.build(sentence)), [sentence])

  return (
    <WidgetFrame
      widgetName="HeadSpecialization"
      label="head specialization — four heads on the same sentence"
      right={<span className="font-mono">each head: hand-designed · causal softmax</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">sentence</span>
          {SENTENCES.map((s, i) => (
            <button
              key={i}
              onClick={() => setSIdx(i)}
              className={cn(
                'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
                sIdx === i
                  ? 'bg-dark-accent text-white'
                  : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
              )}
            >
              #{i + 1}
            </button>
          ))}
          <div className="ml-auto">
            <Readout
              label="subject"
              value={`${sentence.subject} (${sentence.tokens[sentence.subject]})`}
              accent="text-term-green"
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-2 grid-rows-2 gap-3">
        {HEADS.map((h, i) => (
          <HeadPanel key={h.name} spec={h} attn={heads[i]} tokens={sentence.tokens} pos={sentence.pos} />
        ))}
      </div>
    </WidgetFrame>
  )
}

function HeadPanel({
  spec,
  attn,
  tokens,
  pos,
}: {
  spec: HeadSpec
  attn: number[][]
  tokens: string[]
  pos: POS[]
}) {
  const N = tokens.length
  return (
    <div className="rounded border border-dark-border p-2 flex flex-col gap-1 min-h-0 min-w-0 overflow-hidden"
      style={{ backgroundColor: `${spec.hue}0d` }}
    >
      <div className="flex items-center justify-between">
        <span className="text-[10.5px] font-mono uppercase tracking-wider" style={{ color: spec.hue }}>
          {spec.name}
        </span>
        <span className="text-[9.5px] font-mono text-dark-text-disabled italic truncate ml-2">{spec.blurb}</span>
      </div>
      <div className="grid flex-1 gap-[2px] min-h-0" style={{ gridTemplateRows: `auto repeat(${N}, 1fr)`, gridTemplateColumns: `50px repeat(${N}, 1fr)` }}>
        {/* empty corner */}
        <div />
        {/* column labels */}
        {tokens.map((t, j) => (
          <div key={`c-${j}`} className="text-center font-mono text-[8.5px] text-dark-text-muted truncate">
            {t}
          </div>
        ))}
        {/* rows */}
        {attn.map((row, i) => (
          <Fragment key={`r-${i}`}>
            <div className="text-right pr-1 font-mono text-[8.5px] text-dark-text-secondary truncate flex items-center justify-end">
              {tokens[i]}
              <span className="ml-1 text-dark-text-disabled text-[7.5px]">{pos[i]}</span>
            </div>
            {row.map((v, j) => {
              const intensity = 0.06 + Math.min(1, v) * 0.9
              return (
                <div
                  key={`${i}-${j}`}
                  className="rounded-[1px]"
                  style={{ backgroundColor: spec.hue, opacity: intensity }}
                  title={`${tokens[i]} → ${tokens[j]}: ${v.toFixed(2)}`}
                />
              )
            })}
          </Fragment>
        ))}
      </div>
    </div>
  )
}
