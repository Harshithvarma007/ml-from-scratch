'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { SkipBack, StepForward, FastForward } from 'lucide-react'
import { cn } from '@/lib/utils'

// 64 physical KV-cache pages (each holds 16 tokens). Four concurrent requests
// allocate pages as they generate. Paged allocation grabs any free slot; the
// page table maps logical -> physical. Contiguous allocation below reserves
// a large block per request up front, proportional to max length, wasting
// unused slots (internal fragmentation).
//
// "Step" ticks all active requests forward by one token; allocate a fresh
// page when the current one fills. "Run" fast-forwards to completion.

const N_PAGES = 64
const COLS = 16
const PAGE_CAP = 16

const REQUESTS: { id: string; color: string; target: number; maxReserve: number }[] = [
  { id: 'R1', color: '#67e8f9', target: 96,  maxReserve: 128 },
  { id: 'R2', color: '#fbbf24', target: 160, maxReserve: 256 },
  { id: 'R3', color: '#4ade80', target: 72,  maxReserve: 128 },
  { id: 'R4', color: '#a78bfa', target: 192, maxReserve: 256 },
]

type PagedState = {
  tick: number
  pageTable: Record<string, number[]> // rid -> list of physical page indices in logical order
  pageOwner: (string | null)[]         // physical -> rid or null
  generated: Record<string, number>     // rid -> tokens produced
}

function simulatePagedAllUpTo(maxTick: number): PagedState[] {
  const frames: PagedState[] = []
  const pageOwner: (string | null)[] = new Array(N_PAGES).fill(null)
  const pageTable: Record<string, number[]> = {}
  const generated: Record<string, number> = {}
  for (const r of REQUESTS) {
    pageTable[r.id] = []
    generated[r.id] = 0
  }

  frames.push({
    tick: 0,
    pageTable: JSON.parse(JSON.stringify(pageTable)),
    pageOwner: pageOwner.slice(),
    generated: { ...generated },
  })

  // Alternate requests taking turns — each tick, each still-running request
  // appends one token.
  for (let t = 1; t <= maxTick; t++) {
    for (const r of REQUESTS) {
      if (generated[r.id] >= r.target) continue
      // Need a new page?
      const have = pageTable[r.id].length
      const needPages = Math.ceil((generated[r.id] + 1) / PAGE_CAP)
      if (needPages > have) {
        // Allocate a non-contiguous free slot. Use hash-ish spread for visual
        // realism: pick a pseudo-random free slot derived from rid + index.
        const seed = (r.id.charCodeAt(1) * 31 + have * 7) % N_PAGES
        let p = seed
        for (let tries = 0; tries < N_PAGES; tries++) {
          if (pageOwner[p] === null) break
          p = (p + 1) % N_PAGES
        }
        pageOwner[p] = r.id
        pageTable[r.id].push(p)
      }
      generated[r.id] += 1
    }
    frames.push({
      tick: t,
      pageTable: JSON.parse(JSON.stringify(pageTable)),
      pageOwner: pageOwner.slice(),
      generated: { ...generated },
    })
  }
  return frames
}

const MAX_TICK = 200
const FRAMES = simulatePagedAllUpTo(MAX_TICK)

type ContiguousState = {
  start: number
  reserved: number
  used: number
  color: string
}

function simulateContiguousAt(tick: number): ContiguousState[] {
  let cursor = 0
  const out: ContiguousState[] = []
  for (const r of REQUESTS) {
    const reservedPages = Math.ceil(r.maxReserve / PAGE_CAP)
    const usedTokens = Math.min(tick, r.target)
    const usedPages = Math.ceil(usedTokens / PAGE_CAP)
    out.push({ start: cursor, reserved: reservedPages, used: usedPages, color: r.color })
    cursor += reservedPages
  }
  return out
}

export default function KVPaging() {
  const [i, setI] = useState(0)
  const frame = FRAMES[Math.min(i, FRAMES.length - 1)]
  const contig = useMemo(() => simulateContiguousAt(frame.tick), [frame.tick])

  const totalContig = contig.reduce((a, c) => a + c.reserved, 0)
  const usedContig = contig.reduce((a, c) => a + c.used, 0)

  const pagedUsed = frame.pageOwner.filter((p) => p !== null).length

  return (
    <WidgetFrame
      widgetName="KVPaging"
      label="paged KV cache — logical pages → any free physical page"
      right={<span className="font-mono">{N_PAGES} pages · {PAGE_CAP} tokens each · 4 concurrent requests</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <Button onClick={() => setI(0)}>
              <span className="inline-flex items-center gap-1">
                <SkipBack size={11} /> reset
              </span>
            </Button>
            <Button onClick={() => setI(Math.min(FRAMES.length - 1, i + 8))} variant="primary" disabled={i === FRAMES.length - 1}>
              <span className="inline-flex items-center gap-1">
                +8 ticks <StepForward size={11} />
              </span>
            </Button>
            <Button onClick={() => setI(FRAMES.length - 1)} disabled={i === FRAMES.length - 1}>
              <span className="inline-flex items-center gap-1">
                run <FastForward size={11} />
              </span>
            </Button>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="tick" value={String(frame.tick)} />
            <Readout label="paged pages used" value={`${pagedUsed} / ${N_PAGES}`} accent="text-term-green" />
            <Readout
              label="contig reserved"
              value={`${totalContig} (${usedContig} actual)`}
              accent="text-term-rose"
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <div className="grid h-full grid-cols-1 md:grid-cols-[1fr_240px] gap-4 min-h-0">
          {/* Main — paged vs. contiguous stacked */}
          <div className="flex flex-col gap-3 min-h-0">
            <div>
              <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-1">
                paged: physical page frame (non-contiguous)
              </div>
              <PagedGrid pageOwner={frame.pageOwner} />
            </div>

            <div>
              <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-1">
                contiguous: each request reserves its max-length slab
              </div>
              <ContiguousGrid contig={contig} />
              <div className="mt-1 flex flex-wrap gap-2 font-mono text-[9.5px] text-dark-text-disabled">
                <span>filled = used</span>
                <span>hatched = reserved but unused (fragmentation)</span>
              </div>
            </div>
          </div>

          {/* Right: page tables + legend */}
          <div className="flex flex-col gap-2 min-w-0">
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
              page tables (logical → physical)
            </div>
            <div className="flex-1 min-h-0 overflow-auto rounded border border-dark-border bg-dark-bg/60 p-2">
              {REQUESTS.map((r) => {
                const pages = frame.pageTable[r.id] ?? []
                const gen = frame.generated[r.id] ?? 0
                return (
                  <div key={r.id} className="mb-2 font-mono text-[10px]">
                    <div className="flex items-center gap-2 mb-1">
                      <div className="w-2 h-2 rounded-sm" style={{ backgroundColor: r.color }} />
                      <span style={{ color: r.color }}>{r.id}</span>
                      <span className="text-dark-text-disabled">{gen} / {r.target} tok</span>
                    </div>
                    <div className="flex flex-wrap gap-0.5 text-[9.5px]">
                      {pages.length === 0 ? (
                        <span className="text-dark-text-disabled italic">no pages yet</span>
                      ) : (
                        pages.map((p, j) => (
                          <span
                            key={`${j}`}
                            className="px-1 py-[1px] rounded-sm border text-dark-text-muted"
                            style={{ borderColor: r.color, color: r.color, opacity: 0.85 }}
                          >
                            L{j}→P{p}
                          </span>
                        ))
                      )}
                    </div>
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

function PagedGrid({ pageOwner }: { pageOwner: (string | null)[] }) {
  const colorOf = (rid: string | null) => {
    if (!rid) return null
    return REQUESTS.find((r) => r.id === rid)?.color ?? '#555'
  }
  const rows = N_PAGES / COLS
  return (
    <div
      className="grid gap-[2px] rounded border border-dark-border bg-dark-bg/60 p-1"
      style={{
        gridTemplateColumns: `repeat(${COLS}, 1fr)`,
        gridTemplateRows: `repeat(${rows}, 1fr)`,
        aspectRatio: `${COLS} / ${rows}`,
      }}
    >
      {pageOwner.map((rid, idx) => (
        <div
          key={idx}
          title={rid ? `page ${idx} — ${rid}` : `page ${idx} — free`}
          className={cn(
            'rounded-[2px] flex items-center justify-center text-[8px] font-mono',
            rid ? 'text-white/70' : 'text-dark-text-disabled/40',
          )}
          style={{
            backgroundColor: rid ? colorOf(rid)! : '#1a1a1f',
            opacity: rid ? 0.85 : 0.55,
            border: rid ? 'none' : '1px dashed #2a2a32',
          }}
        >
          {idx % 4 === 0 && !rid ? idx : ''}
        </div>
      ))}
    </div>
  )
}

function ContiguousGrid({ contig }: { contig: ContiguousState[] }) {
  const total = N_PAGES
  return (
    <div
      className="grid gap-[2px] rounded border border-dark-border bg-dark-bg/60 p-1"
      style={{ gridTemplateColumns: `repeat(${total}, 1fr)` }}
    >
      {Array.from({ length: total }).map((_, idx) => {
        // Which reservation holds this physical page?
        let owner: ContiguousState | null = null
        let isUsed = false
        for (const c of contig) {
          if (idx >= c.start && idx < c.start + c.reserved) {
            owner = c
            isUsed = idx < c.start + c.used
            break
          }
        }
        return (
          <div
            key={idx}
            className="h-6 rounded-[2px]"
            style={{
              backgroundColor: owner ? (isUsed ? owner.color : '#1a1a1f') : '#0f0f14',
              border: owner && !isUsed ? `1px dashed ${owner.color}` : owner ? 'none' : '1px dashed #2a2a32',
              opacity: owner ? (isUsed ? 0.9 : 0.5) : 0.4,
            }}
            title={owner ? `res'd to ${owner.color} · ${isUsed ? 'used' : 'fragmentation'}` : 'free'}
          />
        )
      })}
    </div>
  )
}
