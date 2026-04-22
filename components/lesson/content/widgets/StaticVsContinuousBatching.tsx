'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Gantt-style chart for 8 requests with varying generation lengths. We model
// a batch slot of 8 concurrent streams. In "static" batching the whole batch
// runs until the longest sequence finishes; finished slots emit wasted GPU
// ticks (gray). In "continuous" batching a newly arriving request slots into
// any freed lane the same step. Toggle between the two modes.

const SLOTS = 8

// Scripted requests with arrival tick + length. Under static we start all 8
// at t=0 and waste the tails. Under continuous, arrivals 9..12 drop in as
// earlier streams finish.
const REQUESTS = [
  { id: 'R1', arrive: 0, len: 12, color: '#67e8f9' },
  { id: 'R2', arrive: 0, len: 6,  color: '#fbbf24' },
  { id: 'R3', arrive: 0, len: 20, color: '#4ade80' },
  { id: 'R4', arrive: 0, len: 8,  color: '#a78bfa' },
  { id: 'R5', arrive: 0, len: 14, color: '#fb7185' },
  { id: 'R6', arrive: 0, len: 5,  color: '#f472b6' },
  { id: 'R7', arrive: 0, len: 18, color: '#fb923c' },
  { id: 'R8', arrive: 0, len: 10, color: '#818cf8' },
  // Incoming queue (only continuous mode pulls these in)
  { id: 'R9',  arrive: 6,  len: 9,  color: '#5eead4' },
  { id: 'R10', arrive: 9,  len: 11, color: '#e879f9' },
  { id: 'R11', arrive: 12, len: 7,  color: '#34d399' },
  { id: 'R12', arrive: 15, len: 6,  color: '#fde68a' },
]

const T_MAX = 22

type Cell = { rid: string; color: string; active: boolean }

function simulateStatic(): Cell[][] {
  // slots[t][lane] = cell
  const schedule: Cell[][] = []
  const batch = REQUESTS.slice(0, SLOTS)
  const maxLen = Math.max(...batch.map((r) => r.len))
  for (let t = 0; t < maxLen; t++) {
    const row: Cell[] = []
    for (let s = 0; s < SLOTS; s++) {
      const req = batch[s]
      if (t < req.len) row.push({ rid: req.id, color: req.color, active: true })
      else row.push({ rid: '—', color: '#2a2a32', active: false })
    }
    schedule.push(row)
  }
  return schedule
}

function simulateContinuous(): Cell[][] {
  // Start with the 8 initial requests. Whenever a lane frees up, pull from
  // the queue of arrivals whose arrive-tick ≤ current t.
  const lanes: { req: typeof REQUESTS[number]; remaining: number }[] = REQUESTS.slice(0, SLOTS).map(
    (r) => ({ req: r, remaining: r.len }),
  )
  const queue = REQUESTS.slice(SLOTS).map((r) => ({ ...r }))

  const schedule: Cell[][] = []
  for (let t = 0; t < T_MAX; t++) {
    // Admit any queue arrivals into free lanes.
    for (let s = 0; s < SLOTS; s++) {
      if (!lanes[s] || lanes[s].remaining <= 0) {
        const pickIdx = queue.findIndex((q) => q.arrive <= t)
        if (pickIdx >= 0) {
          const q = queue.splice(pickIdx, 1)[0]
          lanes[s] = { req: q, remaining: q.len }
        }
      }
    }
    const row: Cell[] = []
    for (let s = 0; s < SLOTS; s++) {
      const slot = lanes[s]
      if (slot && slot.remaining > 0) {
        row.push({ rid: slot.req.id, color: slot.req.color, active: true })
        slot.remaining -= 1
      } else {
        row.push({ rid: '—', color: '#2a2a32', active: false })
      }
    }
    schedule.push(row)
  }
  return schedule
}

export default function StaticVsContinuousBatching() {
  const [mode, setMode] = useState<'static' | 'continuous'>('static')

  const staticSched = useMemo(simulateStatic, [])
  const contSched = useMemo(simulateContinuous, [])
  const sched = mode === 'static' ? staticSched : contSched

  const wastedTicks = sched.reduce(
    (acc, row) => acc + row.filter((c) => !c.active).length,
    0,
  )
  const totalTicks = sched.length * SLOTS
  const activeTicks = totalTicks - wastedTicks
  const utilPct = (activeTicks / totalTicks) * 100

  const completedRequests = useMemo(() => {
    // Count distinct request ids that appear and then disappear (or persist at last tick)
    const ids = new Set<string>()
    for (const row of sched) for (const c of row) if (c.active) ids.add(c.rid)
    return ids.size
  }, [sched])

  return (
    <WidgetFrame
      widgetName="StaticVsContinuousBatching"
      label="static vs. continuous batching — lanes × time"
      right={<span className="font-mono">{SLOTS} slots · {sched.length} ticks · gray = wasted GPU time</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setMode('static')}
              className={cn(
                'px-3 py-1 rounded text-[11px] font-mono uppercase tracking-wider transition-all border',
                mode === 'static'
                  ? 'border-term-rose text-term-rose bg-term-rose/10'
                  : 'border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
              )}
            >
              static
            </button>
            <button
              onClick={() => setMode('continuous')}
              className={cn(
                'px-3 py-1 rounded text-[11px] font-mono uppercase tracking-wider transition-all border',
                mode === 'continuous'
                  ? 'border-term-green text-term-green bg-term-green/10'
                  : 'border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
              )}
            >
              continuous
            </button>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="wasted" value={`${wastedTicks}`} accent="text-term-rose" />
            <Readout label="utilization" value={`${utilPct.toFixed(0)}%`} accent={utilPct > 90 ? 'text-term-green' : 'text-term-amber'} />
            <Readout label="requests served" value={String(completedRequests)} accent="text-term-cyan" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <div className="grid h-full grid-cols-1 md:grid-cols-[80px_1fr] grid-rows-[auto_1fr_auto] gap-x-3 gap-y-2 min-h-0">
          {/* Header */}
          <div />
          <div className="grid gap-[2px] font-mono text-[9px] text-dark-text-disabled" style={{ gridTemplateColumns: `repeat(${sched.length}, 1fr)` }}>
            {Array.from({ length: sched.length }).map((_, t) => (
              <div key={t} className="text-center">
                {t % 2 === 0 ? t : ''}
              </div>
            ))}
          </div>

          {/* Lane labels */}
          <div className="flex flex-col justify-around font-mono text-[10px] text-dark-text-muted pr-2">
            {Array.from({ length: SLOTS }).map((_, s) => (
              <div key={s}>slot {s}</div>
            ))}
          </div>

          {/* Gantt grid */}
          <div className="min-h-0 rounded border border-dark-border bg-dark-bg/60 p-1 overflow-hidden">
            <div
              className="grid gap-[1px] h-full"
              style={{
                gridTemplateColumns: `repeat(${sched.length}, 1fr)`,
                gridTemplateRows: `repeat(${SLOTS}, 1fr)`,
              }}
            >
              {/* We need to iterate row-major by slot first for gantt layout */}
              {Array.from({ length: SLOTS }).flatMap((_, s) =>
                sched.map((row, t) => {
                  const cell = row[s]
                  return (
                    <div
                      key={`${s}-${t}`}
                      title={`slot ${s} · t=${t} · ${cell.rid}`}
                      className="rounded-[1px]"
                      style={{
                        backgroundColor: cell.active ? cell.color : '#1a1a1f',
                        opacity: cell.active ? 0.85 : 0.55,
                        border: cell.active ? 'none' : '1px dashed rgba(247, 113, 133, 0.2)',
                      }}
                    />
                  )
                }),
              )}
            </div>
          </div>

          {/* Legend */}
          <div />
          <div className="flex flex-wrap gap-2 font-mono text-[10px]">
            {REQUESTS.slice(0, mode === 'static' ? SLOTS : REQUESTS.length).map((r) => (
              <div key={r.id} className="flex items-center gap-1">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: r.color, opacity: 0.85 }} />
                <span className="text-dark-text-muted">{r.id}</span>
                <span className="text-dark-text-disabled">len={r.len}</span>
              </div>
            ))}
            <div className="flex items-center gap-1 ml-2">
              <div className="w-3 h-3 rounded-sm border border-dashed border-term-rose/40" />
              <span className="text-term-rose">wasted</span>
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
