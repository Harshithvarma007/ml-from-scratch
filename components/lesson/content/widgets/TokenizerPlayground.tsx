'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Four tokenization strategies applied to the same input: whitespace, char,
// naive BPE (with a tiny hand-authored merge table), and word-level. Tokens
// are shown as colored chips with ids so the viewer can compare granularity
// and compression ratio side-by-side.

type Strategy = 'whitespace' | 'char' | 'bpe' | 'word'

const SAMPLES: readonly string[] = [
  'the quick brown fox jumps over the lazy dog',
  'tokenization chops text into pieces',
  'unbelievable lowercase text like this works well',
]

// A small hand-picked BPE merge table, learned in advance from toy data.
// Ordered: earlier merges take priority. The tokenizer repeatedly applies the
// first applicable merge across the whole sequence until none apply.
const BPE_MERGES: readonly (readonly [string, string])[] = [
  ['t', 'h'],     // th
  ['i', 'n'],     // in
  ['o', 'n'],     // on
  ['e', 'r'],     // er
  ['a', 't'],     // at
  ['th', 'e'],    // the
  ['o', 'w'],     // ow
  ['j', 'u'],     // ju
  ['ju', 'm'],    // jum
  ['jum', 'p'],   // jump
  ['l', 'a'],     // la
  ['la', 'z'],    // laz
  ['laz', 'y'],   // lazy
  ['on', 'g'],    // ong
  ['in', 'g'],    // ing
  ['ow', 'n'],    // own
  ['f', 'o'],     // fo
  ['fo', 'x'],    // fox
  ['d', 'o'],     // do
  ['do', 'g'],    // dog
  ['t', 'e'],     // te
  ['te', 'x'],    // tex
  ['tex', 't'],   // text
  ['l', 'i'],     // li
  ['li', 'k'],    // lik
  ['lik', 'e'],   // like
  ['l', 'o'],     // lo
  ['lo', 'w'],    // low
  ['at', 'i'],    // ati
  ['ati', 'on'],  // ation
]

function whitespaceTokens(text: string): string[] {
  return text.trim().split(/\s+/).filter(Boolean)
}

function charTokens(text: string): string[] {
  return Array.from(text.replace(/\s+/g, '_'))
}

function wordTokens(text: string): string[] {
  // Word-level: punctuation split off as separate tokens.
  return text.match(/[A-Za-z0-9]+|[^\sA-Za-z0-9]/g) ?? []
}

function bpeOneWord(word: string): string[] {
  let toks = Array.from(word)
  let changed = true
  while (changed) {
    changed = false
    for (const [a, b] of BPE_MERGES) {
      const next: string[] = []
      let i = 0
      let touched = false
      while (i < toks.length) {
        if (i < toks.length - 1 && toks[i] === a && toks[i + 1] === b) {
          next.push(a + b)
          i += 2
          touched = true
        } else {
          next.push(toks[i])
          i++
        }
      }
      if (touched) {
        toks = next
        changed = true
        break
      }
    }
  }
  return toks
}

function bpeTokens(text: string): string[] {
  const words = text.toLowerCase().split(/\s+/).filter(Boolean)
  const out: string[] = []
  words.forEach((w, idx) => {
    const sub = bpeOneWord(w)
    sub.forEach((s, i) => out.push(i === 0 ? s : s))
    if (idx < words.length - 1) out.push('_')
  })
  return out
}

// Deterministic id per unique string. Collisions are fine — we just use it to
// give each token a stable numeric label so the chip looks like a real id.
function stableId(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return (h >>> 0) % 50000
}

const STRATEGY_COLORS: Record<Strategy, string> = {
  whitespace: '#67e8f9',
  char: '#f472b6',
  bpe: '#fbbf24',
  word: '#4ade80',
}

const STRATEGY_DESC: Record<Strategy, string> = {
  whitespace: 'split on spaces — the simplest thing that works',
  char: 'each character is a token — longest sequences, tiniest vocab',
  bpe: 'byte-pair encoding with a hand-authored merge table',
  word: 'word-level with punctuation split off separately',
}

function tokenize(text: string, s: Strategy): string[] {
  if (s === 'whitespace') return whitespaceTokens(text)
  if (s === 'char') return charTokens(text)
  if (s === 'bpe') return bpeTokens(text)
  return wordTokens(text)
}

export default function TokenizerPlayground() {
  const [text, setText] = useState(SAMPLES[0])

  const results = useMemo(() => {
    const strategies: Strategy[] = ['whitespace', 'word', 'bpe', 'char']
    return strategies.map((s) => {
      const tokens = tokenize(text, s)
      const chars = text.length
      const ratio = tokens.length > 0 ? chars / tokens.length : 0
      return { strategy: s, tokens, count: tokens.length, ratio }
    })
  }, [text])

  return (
    <WidgetFrame
      widgetName="TokenizerPlayground"
      label="tokenizer playground — four ways to chop the same sentence"
      right={<span className="font-mono">compression ratio = chars / tokens (higher = coarser)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">quick-picks:</span>
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
              &quot;{s.slice(0, 28)}{s.length > 28 ? '…' : ''}&quot;
            </button>
          ))}
          <Readout label="chars" value={String(text.length)} accent="text-term-amber" />
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

        <div className="grid grid-cols-2 gap-3 flex-1 min-h-0">
          {results.map(({ strategy, tokens, count, ratio }) => (
            <div
              key={strategy}
              className="flex flex-col gap-1.5 bg-dark-surface-elevated/30 rounded border border-dark-border p-2.5 overflow-hidden"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: STRATEGY_COLORS[strategy] }}
                  />
                  <span className="font-mono text-[11px] uppercase tracking-wider" style={{ color: STRATEGY_COLORS[strategy] }}>
                    {strategy}
                  </span>
                </div>
                <div className="flex items-center gap-3 font-mono text-[10px] tabular-nums">
                  <span className="text-dark-text-muted">n = <span className="text-dark-text-primary">{count}</span></span>
                  <span className="text-dark-text-muted">ratio = <span className="text-dark-text-primary">{ratio.toFixed(2)}</span></span>
                </div>
              </div>
              <div className="text-[9.5px] font-mono text-dark-text-disabled">{STRATEGY_DESC[strategy]}</div>
              <div className="flex flex-wrap gap-1 overflow-auto pt-0.5">
                {tokens.map((t, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 font-mono text-[10.5px]"
                    style={{
                      backgroundColor: `${STRATEGY_COLORS[strategy]}22`,
                      border: `1px solid ${STRATEGY_COLORS[strategy]}55`,
                      color: STRATEGY_COLORS[strategy],
                    }}
                  >
                    <span>{t === ' ' ? '␣' : t === '\n' ? '↵' : t}</span>
                    <span className="text-dark-text-disabled text-[9px] tabular-nums">{stableId(t)}</span>
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </WidgetFrame>
  )
}
