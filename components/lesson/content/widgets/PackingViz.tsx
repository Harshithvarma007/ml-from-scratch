'use client'

// 12 variable-length documents packed into fixed-length rows. Top half:
// "no packing" — each doc gets its own row and the tail is padding. Bottom
// half: "greedy packing" — docs are concatenated until they overflow, then
// start a new row. A side-by-side padding-efficiency readout hammers home
// why packing matters.

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// 12 docs with a long-tailed length distribution (some tiny, a few near-full)
function buildDocs(seed: number, ctxLen: number): number[] {
  const rng = mulberry32(seed)
  const out: number[] = []
  for (let i = 0; i < 12; i++) {
    // mix of short, medium, long — clipped to ctx
    const r = rng()
    let len: number
    if (r < 0.4) len = 5 + Math.floor(rng() * 25)
    else if (r < 0.8) len = 30 + Math.floor(rng() * 60)
    else len = 80 + Math.floor(rng() * 50)
    out.push(Math.min(ctxLen, Math.max(3, len)))
  }
  return out
}

const COLORS = ['#fbbf24', '#67e8f9', '#a78bfa', '#f472b6', '#4ade80', '#fb923c', '#5eead4', '#f87171', '#c084fc', '#60a5fa', '#fde047', '#22d3ee']

type Row = { segments: { doc: number; len: number }[]; used: number; pad: number }

function packRows(docs: number[], ctxLen: number, greedy: boolean): Row[] {
  if (!greedy) {
    return docs.map((len, i) => ({
      segments: [{ doc: i, len }],
      used: len,
      pad: ctxLen - len,
    }))
  }
  // greedy: fill each row until the next doc doesn't fit
  const rows: Row[] = []
  let cur: Row = { segments: [], used: 0, pad: ctxLen }
  for (let i = 0; i < docs.length; i++) {
    const len = docs[i]
    if (cur.used + len > ctxLen) {
      // seal current
      cur.pad = ctxLen - cur.used
      rows.push(cur)
      cur = { segments: [], used: 0, pad: ctxLen }
    }
    cur.segments.push({ doc: i, len })
    cur.used += len
  }
  if (cur.segments.length > 0) {
    cur.pad = ctxLen - cur.used
    rows.push(cur)
  }
  return rows
}

function stats(rows: Row[], ctxLen: number): { used: number; pad: number; efficiency: number } {
  let used = 0
  let pad = 0
  for (const r of rows) {
    used += r.used
    pad += r.pad
  }
  const efficiency = used / Math.max(1, used + pad)
  return { used, pad, efficiency }
}

export default function PackingViz() {
  const [ctxLen, setCtxLen] = useState(128)
  const [seed, setSeed] = useState(1)

  const docs = useMemo(() => buildDocs(seed, ctxLen), [seed, ctxLen])
  const noPack = useMemo(() => packRows(docs, ctxLen, false), [docs, ctxLen])
  const pack = useMemo(() => packRows(docs, ctxLen, true), [docs, ctxLen])

  const noPackStats = stats(noPack, ctxLen)
  const packStats = stats(pack, ctxLen)

  return (
    <WidgetFrame
      widgetName="PackingViz"
      label="sequence packing — no packing vs greedy"
      right={
        <span className="font-mono">
          12 docs · context {ctxLen} · rows: {noPack.length} → {pack.length}
        </span>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-3 flex-1 min-w-[240px] font-mono text-[12px]">
            <span className="text-dark-text-secondary whitespace-nowrap">context_len</span>
            <input
              type="range"
              min={48}
              max={256}
              step={16}
              value={ctxLen}
              onChange={(e) => setCtxLen(Number(e.target.value))}
              className="flex-1 h-1 rounded-full bg-dark-border cursor-pointer accent-term-cyan"
            />
            <span className="text-dark-text-primary tabular-nums w-10 text-right">{ctxLen}</span>
          </label>
          <button
            onClick={() => setSeed((s) => s + 1)}
            className="px-3 py-1 rounded-md text-[11px] font-mono uppercase tracking-wider border border-dark-border text-dark-text-secondary hover:text-dark-text-primary hover:border-dark-border-hover"
          >
            reshuffle docs
          </button>
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="no pack"
              value={`${(noPackStats.efficiency * 100).toFixed(0)}% used`}
              accent="text-term-rose"
            />
            <Readout
              label="greedy pack"
              value={`${(packStats.efficiency * 100).toFixed(0)}% used`}
              accent="text-term-green"
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-rows-2 gap-4 overflow-hidden">
        <PackView
          title="no packing — one doc per row, rest is padding"
          rows={noPack}
          ctxLen={ctxLen}
          accent="text-term-rose"
          s={noPackStats}
        />
        <PackView
          title="greedy packing — fill to context, then new row"
          rows={pack}
          ctxLen={ctxLen}
          accent="text-term-green"
          s={packStats}
        />
      </div>
    </WidgetFrame>
  )
}

function PackView({
  title,
  rows,
  ctxLen,
  accent,
  s,
}: {
  title: string
  rows: Row[]
  ctxLen: number
  accent: string
  s: { used: number; pad: number; efficiency: number }
}) {
  return (
    <div className="flex flex-col gap-2 min-h-0 overflow-hidden">
      <div className="flex items-center justify-between text-[10px] font-mono uppercase tracking-wider">
        <span className={accent}>{title}</span>
        <span className="text-dark-text-disabled tabular-nums">
          {rows.length} rows · {s.used} tokens used · {s.pad} padding · {(s.efficiency * 100).toFixed(1)}%
        </span>
      </div>
      <div className="flex-1 min-h-0 overflow-auto flex flex-col gap-[3px] p-2 bg-dark-surface-elevated/20 border border-dark-border rounded">
        {rows.map((row, ri) => (
          <div
            key={ri}
            className="relative h-4 flex rounded-sm overflow-hidden"
            title={`row ${ri} · ${row.used} used · ${row.pad} pad`}
          >
            {row.segments.map((seg, si) => (
              <div
                key={si}
                className="h-full relative group"
                style={{
                  width: `${(seg.len / ctxLen) * 100}%`,
                  backgroundColor: COLORS[seg.doc % COLORS.length],
                  opacity: 0.85,
                  boxShadow: si > 0 ? 'inset 2px 0 0 rgba(0,0,0,0.5)' : undefined,
                }}
              >
                {seg.len > 12 && (
                  <span className="absolute inset-0 flex items-center justify-center text-[9px] font-mono text-dark-bg">
                    #{seg.doc}·{seg.len}
                  </span>
                )}
              </div>
            ))}
            {row.pad > 0 && (
              <div
                className="h-full bg-dark-bg border-l border-dark-border/40 relative"
                style={{ width: `${(row.pad / ctxLen) * 100}%` }}
                title={`${row.pad} padding tokens`}
              >
                {row.pad > 16 && (
                  <span className="absolute inset-0 flex items-center justify-center text-[9px] font-mono text-dark-text-disabled">
                    pad ×{row.pad}
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
