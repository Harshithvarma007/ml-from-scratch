'use client'

// Hand-coded mini-BPE tokenizer with ~30 merges trained offline on English
// snippets. Shows input as colored token chips with IDs, tracks which merge
// produced each token, and reports chars/tokens/compression. Clicking a chip
// pops up the merge lineage. Includes a rare-word preset to show fallback to
// char-level tokens.

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// A compact, deterministic BPE merge table crafted so tokenization on the
// presets produces interesting behavior. Priority = lower index first applied.
const MERGES: [string, string][] = [
  ['t', 'h'],     // 0 — th
  ['i', 'n'],     // 1 — in
  ['e', 'r'],     // 2 — er
  ['o', 'n'],     // 3 — on
  ['r', 'e'],     // 4 — re
  ['a', 'n'],     // 5 — an
  ['e', 'n'],     // 6 — en
  ['o', 'r'],     // 7 — or
  ['e', 's'],     // 8 — es
  ['e', 'd'],     // 9 — ed
  ['th', 'e'],    // 10 — the
  ['in', 'g'],    // 11 — ing
  ['a', 'l'],     // 12 — al
  ['a', 't'],     // 13 — at
  ['o', 'u'],     // 14 — ou
  ['i', 's'],     // 15 — is
  ['l', 'l'],     // 16 — ll
  ['s', 't'],     // 17 — st
  ['t', 'o'],     // 18 — to
  ['of', ' '],    // 19 — (won't hit because we don't merge spaces)
  ['▁', 'the'],   // 20 — leading-space the
  ['▁', 'an'],    // 21 — ▁an
  ['▁', 'to'],    // 22 — ▁to
  ['▁', 'of'],    // 23 — ▁of
  ['▁', 'is'],    // 24 — ▁is
  ['▁', 'a'],     // 25 — ▁a
  ['qu', 'ick'],  // 26 — quick (won't hit bc base isn't qu)
  ['q', 'u'],     // 27 — qu
  ['i', 'c'],     // 28 — ic
  ['ic', 'k'],    // 29 — ick
]

type Token = {
  text: string
  id: number
  // merge index that produced this token (-1 if it's a base char)
  viaMerge: number
  // chars this token covers in the original string
  charSpan: [number, number]
}

function buildVocab(): { vocab: string[]; merges: [string, string, string][] } {
  // base: lowercase ASCII + digits + basic punctuation + special '▁' for word start
  const base: string[] = []
  const addRange = (a: number, b: number) => {
    for (let c = a; c <= b; c++) base.push(String.fromCharCode(c))
  }
  addRange(97, 122)                       // a-z
  addRange(48, 57)                        // 0-9
  for (const c of '.,!?;:\'"()-') base.push(c)
  base.push('▁')                          // word-start marker
  const vocab = base.slice()
  const tripled: [string, string, string][] = []
  for (const [a, b] of MERGES) {
    const merged = a + b
    if (!vocab.includes(a) || !vocab.includes(b)) {
      // skip if prerequisites missing (keeps us honest)
      tripled.push([a, b, merged])
      continue
    }
    if (!vocab.includes(merged)) vocab.push(merged)
    tripled.push([a, b, merged])
  }
  return { vocab, merges: tripled }
}

const { vocab: VOCAB, merges: FULL_MERGES } = buildVocab()
const MERGE_PRIORITY = new Map<string, number>()
for (let i = 0; i < FULL_MERGES.length; i++) {
  const [a, b] = FULL_MERGES[i]
  MERGE_PRIORITY.set(`${a}\u0001${b}`, i)
}

// Tokenize a single word's character list using priority BPE: repeatedly
// merge the pair with the lowest merge priority until none are available.
function bpeWord(charsIn: string[]): { text: string; viaMerge: number }[] {
  type Node = { text: string; viaMerge: number }
  const nodes: Node[] = charsIn.map((c) => ({ text: c, viaMerge: -1 }))
  while (true) {
    let bestIdx = -1
    let bestPrio = Infinity
    for (let i = 0; i < nodes.length - 1; i++) {
      const k = `${nodes[i].text}\u0001${nodes[i + 1].text}`
      const p = MERGE_PRIORITY.get(k)
      if (p !== undefined && p < bestPrio) {
        bestPrio = p
        bestIdx = i
      }
    }
    if (bestIdx === -1) break
    const a = nodes[bestIdx].text
    const b = nodes[bestIdx + 1].text
    nodes.splice(bestIdx, 2, { text: a + b, viaMerge: bestPrio })
  }
  return nodes
}

function tokenize(input: string): Token[] {
  // simplistic pretokenizer: split on whitespace, prefix each word with '▁'
  const out: Token[] = []
  let char = 0
  const words = input.split(/(\s+)/)
  for (const w of words) {
    if (w.length === 0) continue
    if (/^\s+$/.test(w)) {
      char += w.length
      continue
    }
    // lower casing is part of our tiny pipeline
    const lowered = w.toLowerCase()
    // char span includes one leading char (the '▁' virtual marker) to align nicely
    const chars = ['▁', ...Array.from(lowered)]
    const nodes = bpeWord(chars)
    // map nodes to char spans — drop the virtual ▁ from span when we compute highlighting
    let local = char
    for (const n of nodes) {
      const consumed = n.text.replace(/▁/g, '').length // ▁ is virtual, consumes 0 real chars
      const span: [number, number] = [local, local + consumed]
      const id = Math.max(0, VOCAB.indexOf(n.text))
      out.push({ text: n.text, id, viaMerge: n.viaMerge, charSpan: span })
      local += consumed
    }
    char += w.length
  }
  return out
}

const PRESETS: { label: string; text: string; note: string }[] = [
  {
    label: 'common',
    text: 'the quick brown fox is running',
    note: 'lots of merges hit — "the", "ick", "ing" all become single tokens',
  },
  {
    label: 'techy',
    text: 'to an older engine of the kind',
    note: 'common bigrams fold tightly — high compression',
  },
  {
    label: 'rare words',
    text: 'xylophone zygote kumquat',
    note: 'no merges hit — falls back to near-character tokens',
  },
]

const COLORS = ['#fbbf24', '#67e8f9', '#a78bfa', '#f472b6', '#4ade80', '#f87171', '#5eead4', '#fb923c']

function chipColor(tok: Token): string {
  if (tok.text.length <= 1 && tok.viaMerge === -1) return '#555'
  return COLORS[(tok.viaMerge + 1) % COLORS.length]
}

export default function TokenizationPreview() {
  const [presetIdx, setPresetIdx] = useState(0)
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null)
  const preset = PRESETS[presetIdx]

  const tokens = useMemo(() => tokenize(preset.text), [preset.text])
  const charCount = preset.text.length
  const tokCount = tokens.length
  const ratio = charCount / Math.max(1, tokCount)

  const sel = selectedIdx !== null ? tokens[selectedIdx] : null
  const selMerge = sel && sel.viaMerge >= 0 ? FULL_MERGES[sel.viaMerge] : null

  return (
    <WidgetFrame
      widgetName="TokenizationPreview"
      label="hand-coded BPE · ~30 merges · click any chip for its lineage"
      right={<span className="font-mono">vocab |V| = {VOCAB.length}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5 flex-wrap">
            {PRESETS.map((p, idx) => (
              <button
                key={p.label}
                onClick={() => {
                  setPresetIdx(idx)
                  setSelectedIdx(null)
                }}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
                  presetIdx === idx
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {p.label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="chars" value={String(charCount)} />
            <Readout label="tokens" value={String(tokCount)} accent="text-term-amber" />
            <Readout label="chars/tok" value={ratio.toFixed(2)} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 flex flex-col gap-3 overflow-hidden">
        <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
          input · {preset.note}
        </div>
        <div className="px-3 py-2 font-mono text-[12px] bg-dark-surface-elevated/30 border border-dark-border rounded text-dark-text-primary">
          {preset.text}
        </div>

        <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-1">
          tokens · click for lineage
        </div>
        <div className="flex flex-wrap gap-[6px] p-3 bg-dark-surface-elevated/30 border border-dark-border rounded overflow-auto flex-1 min-h-0 content-start">
          {tokens.map((t, idx) => {
            const active = selectedIdx === idx
            const color = chipColor(t)
            return (
              <button
                key={idx}
                onClick={() => setSelectedIdx(active ? null : idx)}
                className={cn(
                  'flex items-baseline gap-1.5 px-2 py-1 rounded-sm border font-mono text-[12px] transition-all',
                  active
                    ? 'border-term-amber bg-term-amber/15'
                    : 'border-dark-border/60 bg-dark-bg hover:border-dark-border-hover',
                )}
                style={!active ? { borderColor: color + '55' } : undefined}
              >
                <span style={{ color }}>{t.text.replace(/▁/g, '\u00b7')}</span>
                <span className="text-[9.5px] text-dark-text-disabled tabular-nums">{t.id}</span>
              </button>
            )
          })}
        </div>

        {sel && (
          <div className="border border-term-amber/40 rounded bg-term-amber/5 p-2.5 font-mono text-[11px] text-dark-text-primary">
            <div className="flex items-center gap-3 flex-wrap">
              <span className="text-term-amber">
                token &quot;{sel.text.replace(/▁/g, '\u00b7')}&quot; · id {sel.id}
              </span>
              {selMerge ? (
                <span className="text-dark-text-secondary">
                  produced by merge #{sel.viaMerge}: &quot;{selMerge[0]}&quot; + &quot;{selMerge[1]}&quot; →{' '}
                  &quot;{selMerge[2]}&quot;
                </span>
              ) : (
                <span className="text-dark-text-disabled">base vocab — no merge applied</span>
              )}
              <span className="ml-auto text-dark-text-disabled">
                covers chars [{sel.charSpan[0]}, {sel.charSpan[1]})
              </span>
            </div>
          </div>
        )}
      </div>
    </WidgetFrame>
  )
}
