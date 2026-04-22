'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// 8x8 grid of "patches" with a hand-designed scene. Click a patch to set it
// as the query. Attention weights are a weighted combo of: (a) spatial
// locality (nearby patches attend more), (b) content similarity (patches in
// the same region attend to each other). Toggle between heads to switch the
// pattern (local vs content).

const GRID = 8

// Assign a "region tag" to each patch: sky, ground, object, sun
type Region = 'sky' | 'ground' | 'object' | 'sun'
function regionOf(r: number, c: number): Region {
  if (r < 2 && c > 5) return 'sun'
  if (r < 4) return 'sky'
  if (r === 4 && c >= 2 && c <= 5) return 'object'
  return 'ground'
}

const REGION_COLOR: Record<Region, string> = {
  sky: 'rgba(96, 165, 250, 0.5)',
  ground: 'rgba(139, 92, 40, 0.5)',
  object: 'rgba(167, 139, 250, 0.7)',
  sun: 'rgba(251, 191, 36, 0.8)',
}

type Head = 'local' | 'content' | 'global'

function attention(queryR: number, queryC: number, head: Head): number[][] {
  const W: number[][] = Array.from({ length: GRID }, () => new Array(GRID).fill(0))
  const queryRegion = regionOf(queryR, queryC)
  let total = 0
  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) {
      let score = 0
      if (head === 'local') {
        const d = Math.hypot(r - queryR, c - queryC)
        score = Math.exp(-d * 0.6)
      } else if (head === 'content') {
        score = regionOf(r, c) === queryRegion ? 1 : 0.05
      } else {
        score = 1 // uniform global attention
      }
      W[r][c] = score
      total += score
    }
  }
  // Normalize
  for (let r = 0; r < GRID; r++) for (let c = 0; c < GRID; c++) W[r][c] /= total
  return W
}

export default function PatchAttention() {
  const [query, setQuery] = useState({ r: 4, c: 3 })
  const [head, setHead] = useState<Head>('local')
  const W = useMemo(() => attention(query.r, query.c, head), [query, head])
  const maxW = Math.max(...W.flat())

  // Effective receptive field — how many patches have attention >= max/10
  const effective = W.flat().filter((w) => w >= maxW / 10).length

  return (
    <WidgetFrame
      widgetName="PatchAttention"
      label="patch attention — what a ViT head looks at"
      right={<span className="font-mono">8×8 patch grid · click to pick the query</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            {(['local', 'content', 'global'] as Head[]).map((h) => (
              <button
                key={h}
                onClick={() => setHead(h)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono uppercase transition-all',
                  head === h
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary',
                )}
              >
                {h} head
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="query" value={`(${query.r},${query.c}) · ${regionOf(query.r, query.c)}`} />
            <Readout label="effective RF" value={`${effective} patches`} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 grid grid-cols-[1fr_1fr] gap-8 items-center justify-items-center overflow-auto">
        {/* Left: patch grid with region colors */}
        <div className="flex flex-col items-center gap-2">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            image (regions)
          </div>
          <div className="grid gap-[1px] p-2 rounded bg-dark-bg" style={{ gridTemplateColumns: `repeat(${GRID}, 32px)` }}>
            {Array.from({ length: GRID }).flatMap((_, r) =>
              Array.from({ length: GRID }).map((_, c) => {
                const isQuery = r === query.r && c === query.c
                return (
                  <button
                    key={`${r}-${c}`}
                    onClick={() => setQuery({ r, c })}
                    className={cn(
                      'w-8 h-8 rounded-sm relative border',
                      isQuery ? 'border-term-amber ring-2 ring-term-amber' : 'border-dark-border',
                    )}
                    style={{ backgroundColor: REGION_COLOR[regionOf(r, c)] }}
                  >
                    {isQuery && <span className="absolute inset-0 flex items-center justify-center text-[10px] font-mono text-dark-bg">Q</span>}
                  </button>
                )
              }),
            )}
          </div>
          <div className="flex items-center gap-3 text-[9px] font-mono text-dark-text-disabled">
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: REGION_COLOR.sky }} />sky</span>
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: REGION_COLOR.sun }} />sun</span>
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: REGION_COLOR.object }} />object</span>
            <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: REGION_COLOR.ground }} />ground</span>
          </div>
        </div>

        {/* Right: attention heatmap */}
        <div className="flex flex-col items-center gap-2">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            attention weights (softmax)
          </div>
          <div className="grid gap-[1px] p-2 rounded bg-dark-bg" style={{ gridTemplateColumns: `repeat(${GRID}, 32px)` }}>
            {W.flatMap((row, r) =>
              row.map((w, c) => {
                const intensity = Math.pow(w / maxW, 0.7)
                const isQuery = r === query.r && c === query.c
                return (
                  <div
                    key={`${r}-${c}`}
                    className={cn(
                      'w-8 h-8 rounded-sm border relative',
                      isQuery ? 'border-term-amber' : 'border-transparent',
                    )}
                    style={{ backgroundColor: `rgba(251, 191, 36, ${intensity})` }}
                    title={`w = ${(w * 100).toFixed(1)}%`}
                  >
                    {w > maxW * 0.3 && (
                      <span className="absolute inset-0 flex items-center justify-center text-[8px] font-mono text-dark-bg">
                        {(w * 100).toFixed(0)}
                      </span>
                    )}
                  </div>
                )
              }),
            )}
          </div>
          <div className="text-[10px] font-mono text-dark-text-muted">
            {head === 'local' && 'local head: spatial proximity dominates'}
            {head === 'content' && 'content head: same-region patches attend to each other'}
            {head === 'global' && 'global head: uniform attention over all patches'}
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
