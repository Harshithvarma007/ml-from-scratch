'use client'

// Free-text input inspector. Three aligned rows: raw chars on top, UTF-8 bytes
// in the middle, BPE tokens on the bottom. Click anywhere to highlight the
// corresponding span on every other row. Counts of each level are shown in
// the readout.

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Tiny merge table biased toward common code-like and English substrings.
const MERGES: [string, string][] = [
  ['d', 'e'],       // de
  ['de', 'f'],      // def
  ['f', 'o'],       // fo
  ['fo', 'o'],      // foo
  ['b', 'a'],       // ba
  ['ba', 'r'],      // bar
  ['h', 'e'],       // he
  ['he', 'l'],      // hel
  ['hel', 'l'],     // hell
  ['hell', 'o'],    // hello
  ['(', ')'],       // ()
  [':', ' '],       // ": "
  ['#', ' '],       // "# "
  ['(', 'bar'],     // (bar
  [' ', ' '],       // double space
]
const PRIO = new Map<string, number>()
MERGES.forEach(([a, b], i) => PRIO.set(`${a}\u0001${b}`, i))

// Each token carries a byte-range it covers in the original string.
type Tok = {
  text: string
  byteStart: number
  byteEnd: number
}

function tokenize(text: string): { tokens: Tok[]; bytes: Uint8Array; charBytes: number[][] } {
  const encoder = new TextEncoder()
  const bytes = encoder.encode(text)

  // Map each char to the byte range it occupies.
  const charBytes: number[][] = []
  let bp = 0
  for (const ch of text) {
    const chBytes = encoder.encode(ch)
    const span: number[] = []
    for (let k = 0; k < chBytes.length; k++) span.push(bp + k)
    charBytes.push(span)
    bp += chBytes.length
  }

  // Build initial nodes as single chars (we tokenize on chars, record byte span).
  type Node = { text: string; byteStart: number; byteEnd: number }
  const nodes: Node[] = []
  let ci = 0
  for (const ch of text) {
    const span = charBytes[ci]
    nodes.push({ text: ch, byteStart: span[0], byteEnd: span[span.length - 1] + 1 })
    ci += 1
  }

  // Priority merge loop. Only merges where both sides are ASCII-ish — for
  // non-ASCII we leave tokens as full chars (real byte-BPE would merge bytes,
  // but we keep it instructive here).
  while (true) {
    let bi = -1
    let bp = Infinity
    for (let i = 0; i < nodes.length - 1; i++) {
      const k = `${nodes[i].text}\u0001${nodes[i + 1].text}`
      const p = PRIO.get(k)
      if (p !== undefined && p < bp) {
        bp = p
        bi = i
      }
    }
    if (bi === -1) break
    const a = nodes[bi]
    const b = nodes[bi + 1]
    nodes.splice(bi, 2, {
      text: a.text + b.text,
      byteStart: a.byteStart,
      byteEnd: b.byteEnd,
    })
  }

  return {
    tokens: nodes.map((n) => ({ text: n.text, byteStart: n.byteStart, byteEnd: n.byteEnd })),
    bytes,
    charBytes,
  }
}

const COLORS = ['#fbbf24', '#67e8f9', '#a78bfa', '#f472b6', '#4ade80', '#fb923c', '#5eead4', '#f87171']

export default function TokenSplitInspector() {
  const [text, setText] = useState('def foo(bar):  # hello  ')
  const [hover, setHover] = useState<{ level: 'char' | 'byte' | 'tok'; idx: number } | null>(null)

  const { tokens, bytes, charBytes } = useMemo(() => tokenize(text), [text])

  // Compute which token each byte belongs to
  const byteToTok = useMemo(() => {
    const arr = new Array<number>(bytes.length).fill(-1)
    tokens.forEach((t, i) => {
      for (let b = t.byteStart; b < t.byteEnd; b++) arr[b] = i
    })
    return arr
  }, [bytes, tokens])

  // Given hover, compute which bytes & chars & token are highlighted
  const highlighted = useMemo(() => {
    if (!hover) return { bytes: new Set<number>(), chars: new Set<number>(), toks: new Set<number>() }
    const setBytes = new Set<number>()
    const setChars = new Set<number>()
    const setToks = new Set<number>()
    if (hover.level === 'tok') {
      const t = tokens[hover.idx]
      if (t) {
        for (let b = t.byteStart; b < t.byteEnd; b++) setBytes.add(b)
        setToks.add(hover.idx)
      }
    } else if (hover.level === 'byte') {
      setBytes.add(hover.idx)
      const ti = byteToTok[hover.idx]
      if (ti >= 0) {
        setToks.add(ti)
        const t = tokens[ti]
        for (let b = t.byteStart; b < t.byteEnd; b++) setBytes.add(b)
      }
    } else {
      // char
      setChars.add(hover.idx)
      const bs = charBytes[hover.idx] ?? []
      for (const b of bs) setBytes.add(b)
      if (bs.length > 0) {
        const ti = byteToTok[bs[0]]
        if (ti >= 0) {
          setToks.add(ti)
          const t = tokens[ti]
          for (let bb = t.byteStart; bb < t.byteEnd; bb++) setBytes.add(bb)
        }
      }
    }
    // back-fill chars from bytes
    charBytes.forEach((bs, ci) => {
      if (bs.every((b) => setBytes.has(b))) setChars.add(ci)
    })
    return { bytes: setBytes, chars: setChars, toks: setToks }
  }, [hover, tokens, byteToTok, charBytes])

  const chars = Array.from(text)
  const hasNonAscii = bytes.some((b) => b > 0x7f)

  return (
    <WidgetFrame
      widgetName="TokenSplitInspector"
      label="chars · bytes · tokens — selection-synced"
      right={<span className="font-mono">hover any row to highlight the rest</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-3 flex-1 min-w-0 font-mono text-[12px]">
            <span className="text-dark-text-secondary whitespace-nowrap">text</span>
            <input
              type="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="flex-1 min-w-0 px-2 py-1 rounded bg-dark-bg border border-dark-border text-dark-text-primary font-mono text-[12px] focus:outline-none focus:border-term-cyan"
            />
          </label>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="chars" value={String(chars.length)} accent="text-term-cyan" />
            <Readout label="bytes" value={String(bytes.length)} accent="text-term-pink" />
            <Readout label="tokens" value={String(tokens.length)} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 flex flex-col gap-4 overflow-hidden">
        {/* Row 1: chars */}
        <Section title="chars" note={hasNonAscii ? 'includes multi-byte characters' : 'all ASCII'}>
          <div className="flex flex-wrap gap-[3px]">
            {chars.map((c, i) => {
              const active = highlighted.chars.has(i)
              return (
                <span
                  key={i}
                  onMouseEnter={() => setHover({ level: 'char', idx: i })}
                  onMouseLeave={() => setHover(null)}
                  className={cn(
                    'font-mono text-[12px] px-1.5 py-[2px] rounded-sm border cursor-default',
                    active
                      ? 'border-term-cyan bg-term-cyan/15 text-term-cyan'
                      : 'border-dark-border/60 bg-dark-bg text-dark-text-primary',
                  )}
                >
                  {c === ' ' ? '\u2423' : c}
                </span>
              )
            })}
          </div>
        </Section>

        {/* Row 2: bytes */}
        <Section title="bytes (UTF-8, hex)" note={`${bytes.length} bytes`}>
          <div className="flex flex-wrap gap-[2px]">
            {Array.from(bytes).map((b, i) => {
              const active = highlighted.bytes.has(i)
              return (
                <span
                  key={i}
                  onMouseEnter={() => setHover({ level: 'byte', idx: i })}
                  onMouseLeave={() => setHover(null)}
                  className={cn(
                    'font-mono text-[10px] px-[4px] py-[1px] rounded-sm border cursor-default tabular-nums',
                    active
                      ? 'border-term-pink bg-term-pink/15 text-term-pink'
                      : b > 0x7f
                        ? 'border-term-pink/30 bg-dark-bg text-term-pink/80'
                        : 'border-dark-border/60 bg-dark-bg text-dark-text-muted',
                  )}
                  title={`byte ${i} = 0x${b.toString(16).padStart(2, '0').toUpperCase()}`}
                >
                  {b.toString(16).padStart(2, '0').toUpperCase()}
                </span>
              )
            })}
          </div>
        </Section>

        {/* Row 3: tokens */}
        <Section title="BPE tokens" note={`${tokens.length} tokens · chars/tok ${(chars.length / Math.max(1, tokens.length)).toFixed(2)}`}>
          <div className="flex flex-wrap gap-[4px]">
            {tokens.map((t, i) => {
              const active = highlighted.toks.has(i)
              const color = COLORS[i % COLORS.length]
              const bytesInTok = t.byteEnd - t.byteStart
              return (
                <span
                  key={i}
                  onMouseEnter={() => setHover({ level: 'tok', idx: i })}
                  onMouseLeave={() => setHover(null)}
                  className={cn(
                    'inline-flex items-baseline gap-1 font-mono text-[12px] px-2 py-[2px] rounded-sm border cursor-default',
                    active
                      ? 'border-term-amber bg-term-amber/15'
                      : 'border-dark-border/60 bg-dark-bg',
                  )}
                  style={!active ? { borderColor: color + '55', color } : { color: '#fbbf24' }}
                >
                  <span>{t.text.replace(/ /g, '\u2423')}</span>
                  <span className="text-[9px] text-dark-text-disabled">{bytesInTok}B</span>
                </span>
              )
            })}
          </div>
        </Section>
      </div>
    </WidgetFrame>
  )
}

function Section({
  title,
  note,
  children,
}: {
  title: string
  note?: string
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-col gap-1.5 min-h-0">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
          {title}
        </span>
        {note && <span className="text-[10px] font-mono text-dark-text-disabled">{note}</span>}
      </div>
      <div className="p-2 bg-dark-surface-elevated/30 border border-dark-border rounded overflow-auto">
        {children}
      </div>
    </div>
  )
}
