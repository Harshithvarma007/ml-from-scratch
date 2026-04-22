'use client'

// Tricky strings tokenized three different ways side-by-side: whitespace-only,
// word-level BPE, and byte-level BPE. Each surprises the viewer in its own way
// — leading whitespace becoming its own token, unicode exploding into bytes,
// CamelCase refusing to split. Cycle through presets with the button.

import { useMemo, useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'

type Case = {
  label: string
  text: string
  note: string
  surprise: string
}

const CASES: Case[] = [
  {
    label: 'emoji',
    text: 'hi 🎉 party',
    note: 'a single codepoint emoji',
    surprise: 'byte-level BPE explodes emoji into 4 raw UTF-8 bytes',
  },
  {
    label: 'leading space',
    text: '    leading whitespace',
    note: 'four spaces before text',
    surprise: 'BPE tokenizers treat each leading space as its own token — padding waste',
  },
  {
    label: 'accent',
    text: 'café',
    note: 'é is U+00E9, a 2-byte UTF-8 char',
    surprise: 'byte-level BPE writes é as two bytes; word-level may fall back',
  },
  {
    label: 'camelCase',
    text: 'HelloWorld',
    note: 'no whitespace boundary',
    surprise: 'whitespace tokenizer sees one giant token; BPE may or may not split',
  },
  {
    label: 'japanese',
    text: '日本語',
    note: 'no spaces — 3 CJK chars',
    surprise: 'whitespace gets one token; byte-level BPE produces 9 bytes (3 chars × 3 bytes)',
  },
  {
    label: 'ALLCAPS',
    text: 'HELLO WORLD',
    note: 'uppercase variants',
    surprise: 'often splits differently from lowercase — capital forms are rarer in BPE training',
  },
]

// === Three toy tokenizers ===

// whitespace: split on ' '. single leading spaces collapse into one blob
function tokenizeWhitespace(text: string): string[] {
  if (text === '') return []
  // preserve groups of whitespace as their own tokens so we can SEE "    "
  const parts = text.split(/(\s+)/).filter((x) => x.length > 0)
  return parts
}

// word-level BPE: a tiny curated merge table applied per-word after pretok
const WBPE_MERGES: [string, string][] = [
  ['h', 'i'], ['t', 'h'], ['th', 'e'],
  ['l', 'l'], ['o', 'o'], ['hel', 'lo'], ['hel', 'l'], ['he', 'l'],
  ['w', 'o'], ['wo', 'r'], ['wor', 'l'], ['worl', 'd'],
  ['p', 'a'], ['pa', 'r'], ['par', 't'], ['part', 'y'],
  ['c', 'a'], ['ca', 'f'],
  // capital versions NOT included — forces fallback
]
const WBPE_PRIO = new Map<string, number>()
WBPE_MERGES.forEach(([a, b], i) => WBPE_PRIO.set(`${a}\u0001${b}`, i))

function tokenizeWordBPE(text: string): string[] {
  // pretok: split on whitespace runs, preserving them as tokens with a marker
  const parts = text.split(/(\s+)/).filter((x) => x.length > 0)
  const out: string[] = []
  for (const p of parts) {
    if (/^\s+$/.test(p)) {
      // each whitespace char becomes its own token — this is the surprise
      for (const c of p) out.push(c === ' ' ? '\u2423' : c)
      continue
    }
    // character-level BPE, lowercased so caps fall back to single chars
    const chars = Array.from(p).map((c) => (c.match(/[A-Za-z]/) ? c : c))
    const nodes = chars.slice()
    while (true) {
      let bi = -1
      let bp = Infinity
      for (let i = 0; i < nodes.length - 1; i++) {
        const k = `${nodes[i].toLowerCase()}\u0001${nodes[i + 1].toLowerCase()}`
        const pr = WBPE_PRIO.get(k)
        if (pr !== undefined && pr < bp) {
          bp = pr
          bi = i
        }
      }
      if (bi === -1) break
      nodes.splice(bi, 2, nodes[bi] + nodes[bi + 1])
    }
    // unicode chars that aren't ASCII fall out as <UNK> for this tokenizer
    for (const n of nodes) {
      if (/[^\x00-\x7F]/.test(n)) {
        for (const c of n) {
          if (/[^\x00-\x7F]/.test(c)) out.push('<UNK>')
          else out.push(c)
        }
      } else {
        out.push(n)
      }
    }
  }
  return out
}

// byte-level BPE: operate on UTF-8 bytes. Apply tiny merges on byte-groups,
// always fall back to raw bytes for anything exotic.
function tokenizeByteBPE(text: string): string[] {
  const bytes = new TextEncoder().encode(text)
  const out: string[] = []
  // simplistic strategy: merge runs of ASCII letters into "word" chunks,
  // keep all other bytes as <0xHH>. Preserve leading-space as 'Ġ' token.
  let i = 0
  while (i < bytes.length) {
    const b = bytes[i]
    if (b === 0x20) {
      // single space marker then next word
      out.push('\u0120') // Ġ
      i += 1
      // attach following alphanumerics to this Ġ prefix in the next chunk
      const start = i
      while (i < bytes.length && ((bytes[i] >= 0x41 && bytes[i] <= 0x5a) || (bytes[i] >= 0x61 && bytes[i] <= 0x7a) || (bytes[i] >= 0x30 && bytes[i] <= 0x39))) {
        i += 1
      }
      if (i > start) {
        out[out.length - 1] = '\u0120' + new TextDecoder().decode(bytes.slice(start, i))
      }
      continue
    }
    if ((b >= 0x41 && b <= 0x5a) || (b >= 0x61 && b <= 0x7a)) {
      const start = i
      while (i < bytes.length && ((bytes[i] >= 0x41 && bytes[i] <= 0x5a) || (bytes[i] >= 0x61 && bytes[i] <= 0x7a) || (bytes[i] >= 0x30 && bytes[i] <= 0x39))) {
        i += 1
      }
      out.push(new TextDecoder().decode(bytes.slice(start, i)))
      continue
    }
    // everything else → raw byte
    out.push(`<0x${b.toString(16).padStart(2, '0').toUpperCase()}>`)
    i += 1
  }
  return out
}

const COLORS = ['#fbbf24', '#67e8f9', '#a78bfa', '#f472b6', '#4ade80', '#fb923c', '#5eead4']

function chipColor(tok: string, idx: number): string {
  if (tok === '<UNK>') return '#f87171'
  if (tok.startsWith('<0x')) return '#f472b6'
  if (tok === '\u2423' || tok.startsWith('\u0120')) return '#a78bfa'
  return COLORS[idx % COLORS.length]
}

function displayTok(tok: string): string {
  if (tok === ' ') return '\u2423'
  if (tok === '\u2423') return '\u2423'
  if (tok.startsWith('\u0120')) return '\u00b7' + tok.slice(1)
  return tok
}

export default function EdgeCaseExplorer() {
  const [idx, setIdx] = useState(0)
  const c = CASES[idx]

  const ws = useMemo(() => tokenizeWhitespace(c.text), [c.text])
  const wbpe = useMemo(() => tokenizeWordBPE(c.text), [c.text])
  const bbpe = useMemo(() => tokenizeByteBPE(c.text), [c.text])

  return (
    <WidgetFrame
      widgetName="EdgeCaseExplorer"
      label="tokenization edge cases — 3 tokenizers, same input"
      right={<span className="font-mono">{c.text.length} chars · {new TextEncoder().encode(c.text).length} bytes</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5 flex-wrap">
            {CASES.map((ca, i) => (
              <button
                key={ca.label}
                onClick={() => setIdx(i)}
                className={cn(
                  'px-2 py-1 rounded text-[10.5px] font-mono transition-all',
                  idx === i
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {ca.label}
              </button>
            ))}
          </div>
          <div className="ml-auto">
            <Button onClick={() => setIdx((idx + 1) % CASES.length)} variant="primary">
              <span className="inline-flex items-center gap-1">
                next <ChevronRight size={11} />
              </span>
            </Button>
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 flex flex-col gap-3 overflow-hidden">
        <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
          input
        </div>
        <div className="px-3 py-2 font-mono text-[12px] bg-dark-surface-elevated/30 border border-dark-border rounded text-dark-text-primary whitespace-pre">
          {c.text.replace(/ /g, '\u2423')}
        </div>
        <div className="text-[10.5px] font-mono text-dark-text-muted leading-snug">
          <span className="text-term-amber">{c.label}:</span> {c.note}
        </div>

        <div className="grid grid-cols-3 gap-3 flex-1 min-h-0 overflow-hidden">
          <TokenizerColumn label="whitespace" tokens={ws} accent="text-term-cyan" />
          <TokenizerColumn label="word BPE" tokens={wbpe} accent="text-term-amber" />
          <TokenizerColumn label="byte-level BPE" tokens={bbpe} accent="text-term-purple" />
        </div>

        <div className="border border-term-rose/40 rounded bg-term-rose/5 p-2 font-mono text-[11px]">
          <span className="text-term-rose">surprise: </span>
          <span className="text-dark-text-secondary">{c.surprise}</span>
        </div>
      </div>
    </WidgetFrame>
  )
}

function TokenizerColumn({
  label,
  tokens,
  accent,
}: {
  label: string
  tokens: string[]
  accent: string
}) {
  return (
    <div className="flex flex-col gap-1.5 min-h-0 overflow-hidden">
      <div className="flex items-center justify-between">
        <span className={cn('text-[10px] font-mono uppercase tracking-wider', accent)}>{label}</span>
        <span className="text-[10px] font-mono text-dark-text-disabled tabular-nums">
          {tokens.length} tok
        </span>
      </div>
      <div className="flex-1 min-h-0 overflow-auto p-2 bg-dark-surface-elevated/30 border border-dark-border rounded flex flex-wrap gap-[3px] content-start">
        {tokens.map((t, i) => (
          <span
            key={i}
            className={cn(
              'font-mono text-[11px] px-1.5 py-[2px] rounded-sm border',
              t === '<UNK>' ? 'border-term-rose/50 bg-term-rose/10 text-term-rose' :
                t.startsWith('<0x') ? 'border-term-pink/40 bg-term-pink/10 text-term-pink' :
                  'border-dark-border/60 bg-dark-bg',
            )}
            style={
              t !== '<UNK>' && !t.startsWith('<0x')
                ? { color: chipColor(t, i), borderColor: chipColor(t, i) + '44' }
                : undefined
            }
          >
            {displayTok(t)}
          </span>
        ))}
      </div>
    </div>
  )
}
