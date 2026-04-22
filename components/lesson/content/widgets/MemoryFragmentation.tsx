'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Memory utilization vs. concurrency for contiguous and paged KV caches.
// Contiguous: every request reserves max_seq_len up front. If avg length is
// ~40% of the max, utilization plateaus near 40%.
// Paged: requests allocate only what they need in 16-token pages. Utilization
// stays above 95% regardless of mix.
//
// Second panel: a 4-request scenario grid showing reserved vs. used vs. free
// side-by-side, with numeric readouts for wasted / fragmented / usable.

const MAX_CONCURRENCY = 48
const MAX_SEQ_LEN = 512
const TOTAL_TOKENS = MAX_CONCURRENCY * MAX_SEQ_LEN // capacity in the "contiguous reserves max" world

// Scenario for the side-by-side grid
const SCENARIO = [
  { id: 'R1', color: '#67e8f9', used: 64,  maxReserve: 512 },
  { id: 'R2', color: '#fbbf24', used: 320, maxReserve: 512 },
  { id: 'R3', color: '#4ade80', used: 128, maxReserve: 512 },
  { id: 'R4', color: '#a78bfa', used: 256, maxReserve: 512 },
]

function contiguousUtilization(concurrency: number, avgFrac: number): number {
  // Pool holds `concurrency` slabs of MAX_SEQ_LEN. Used tokens = concurrency * avg.
  // Reserved = concurrency * MAX_SEQ_LEN. Fragmentation = (1 - avgFrac).
  return Math.min(1, avgFrac)
}

function pagedUtilization(concurrency: number, avgFrac: number): number {
  // With 16-token pages, the only waste is the partial last page per request
  // plus some overhead. Empirically 95–98%.
  const avgLen = avgFrac * MAX_SEQ_LEN
  const pagesPerReq = Math.ceil(avgLen / 16)
  const usedPerReq = avgLen
  const reservedPerReq = pagesPerReq * 16
  const perReqUtil = usedPerReq / reservedPerReq
  // Small dampening as concurrency grows (fewer shared headers, but trivial)
  const overhead = 0.01 * Math.min(1, concurrency / MAX_CONCURRENCY)
  return Math.max(0, Math.min(1, perReqUtil - overhead))
}

export default function MemoryFragmentation() {
  const [avgFrac, setAvgFrac] = useState(0.4) // avg sequence length as fraction of max

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const curve = useMemo(() => {
    const out: { n: number; contig: number; paged: number }[] = []
    for (let n = 1; n <= MAX_CONCURRENCY; n++) {
      out.push({
        n,
        contig: contiguousUtilization(n, avgFrac),
        paged: pagedUtilization(n, avgFrac),
      })
    }
    return out
  }, [avgFrac])

  useEffect(() => {
    const canvas = canvasRef.current
    const box = boxRef.current
    if (!canvas || !box) return
    const dpr = window.devicePixelRatio || 1

    const draw = () => {
      const w = box.clientWidth
      const h = box.clientHeight
      canvas.width = w * dpr
      canvas.height = h * dpr
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      const ctx = canvas.getContext('2d')!
      ctx.setTransform(1, 0, 0, 1, 0, 0)
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, w, h)

      const padL = 46
      const padR = 18
      const padT = 20
      const padB = 34
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const xMin = 1
      const xMax = MAX_CONCURRENCY
      const yMin = 0
      const yMax = 1
      const toSx = (n: number) => padL + ((n - xMin) / (xMax - xMin)) * plotW
      const toSy = (v: number) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH

      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.strokeStyle = '#1e1e1e'

      for (let v = 0; v <= 1; v += 0.2) {
        ctx.beginPath()
        ctx.moveTo(padL, toSy(v))
        ctx.lineTo(padL + plotW, toSy(v))
        ctx.stroke()
        ctx.textAlign = 'right'
        ctx.fillText(`${Math.round(v * 100)}%`, padL - 6, toSy(v) + 3)
      }
      for (let n = 1; n <= xMax; n += 8) {
        ctx.textAlign = 'center'
        ctx.fillText(String(n), toSx(n), padT + plotH + 14)
      }
      ctx.fillText('concurrent requests', padL + plotW / 2, padT + plotH + 28)
      ctx.save()
      ctx.translate(14, padT + plotH / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.textAlign = 'center'
      ctx.fillText('utilization', 0, 0)
      ctx.restore()

      // paged line (green)
      ctx.strokeStyle = '#4ade80'
      ctx.lineWidth = 2.4
      ctx.beginPath()
      curve.forEach((p, i) => {
        const x = toSx(p.n)
        const y = toSy(p.paged)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()

      // contig line (rose)
      ctx.strokeStyle = '#fb7185'
      ctx.lineWidth = 2.4
      ctx.beginPath()
      curve.forEach((p, i) => {
        const x = toSx(p.n)
        const y = toSy(p.contig)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()

      // Shaded fragmentation gap
      ctx.fillStyle = 'rgba(247, 113, 133, 0.12)'
      ctx.beginPath()
      curve.forEach((p, i) => {
        const x = toSx(p.n)
        const y = toSy(p.paged)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      for (let i = curve.length - 1; i >= 0; i--) {
        const p = curve[i]
        ctx.lineTo(toSx(p.n), toSy(p.contig))
      }
      ctx.closePath()
      ctx.fill()

      // legend
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#4ade80'
      ctx.textAlign = 'left'
      ctx.fillText('paged — ~97%', padL + 10, padT + 14)
      ctx.fillStyle = '#fb7185'
      ctx.fillText(`contiguous — ${Math.round(contiguousUtilization(16, avgFrac) * 100)}%`, padL + 120, padT + 14)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [curve, avgFrac])

  // Side-by-side scenario metrics
  const total = SCENARIO.length * MAX_SEQ_LEN
  const actualUsed = SCENARIO.reduce((a, r) => a + r.used, 0)
  const wasted = total - actualUsed // contiguous wastes everything unused
  // paged: pages are 16 tokens; over-allocation is ceil(used/16)*16 - used per req
  const pagedReserved = SCENARIO.reduce((a, r) => a + Math.ceil(r.used / 16) * 16, 0)
  const pagedWaste = pagedReserved - actualUsed

  return (
    <WidgetFrame
      widgetName="MemoryFragmentation"
      label="memory utilization — contiguous vs. paged"
      right={<span className="font-mono">avg len = {Math.round(avgFrac * MAX_SEQ_LEN)} / {MAX_SEQ_LEN}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="avg seq fraction"
            value={avgFrac}
            min={0.1}
            max={1.0}
            step={0.02}
            onChange={setAvgFrac}
            format={(v) => `${Math.round(v * 100)}%`}
            accent="accent-term-cyan"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="contiguous util" value={`${Math.round(avgFrac * 100)}%`} accent="text-term-rose" />
            <Readout label="paged util" value={`${Math.round(pagedUtilization(16, avgFrac) * 100)}%`} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <div className="grid h-full grid-cols-1 md:grid-cols-[1fr_300px] gap-4 min-h-0">
          {/* Left: utilization curve */}
          <div ref={boxRef} className="min-h-0 rounded border border-dark-border bg-dark-bg/60">
            <canvas ref={canvasRef} className="w-full h-full block" />
          </div>

          {/* Right: scenario comparison + readouts */}
          <div className="flex flex-col gap-2 min-w-0">
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
              4-request scenario
            </div>

            <div className="rounded border border-dark-border bg-dark-bg/60 p-2 flex flex-col gap-1">
              <div className="text-[9.5px] font-mono text-dark-text-disabled">contiguous</div>
              {SCENARIO.map((r) => (
                <MemoryBar key={`c-${r.id}`} req={r} mode="contig" />
              ))}
              <div className="text-[9.5px] font-mono text-dark-text-disabled mt-1">paged</div>
              {SCENARIO.map((r) => (
                <MemoryBar key={`p-${r.id}`} req={r} mode="paged" />
              ))}
            </div>

            <div className="rounded border border-dark-border bg-dark-surface-elevated/40 p-3 font-mono text-[10.5px] text-dark-text-muted">
              <MetricRow label="total capacity" value={`${total} tok`} />
              <MetricRow label="actually used" value={`${actualUsed} tok`} accent="text-term-green" />
              <MetricRow
                label="contig wasted"
                value={`${wasted} tok (${Math.round((wasted / total) * 100)}%)`}
                accent="text-term-rose"
              />
              <MetricRow
                label="paged wasted"
                value={`${pagedWaste} tok (${Math.round((pagedWaste / pagedReserved) * 100)}%)`}
                accent="text-term-amber"
              />
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function MemoryBar({
  req,
  mode,
}: {
  req: (typeof SCENARIO)[number]
  mode: 'contig' | 'paged'
}) {
  const usedFrac = req.used / MAX_SEQ_LEN
  const pagedReservedFrac = (Math.ceil(req.used / 16) * 16) / MAX_SEQ_LEN
  return (
    <div className="flex items-center gap-2 font-mono text-[10px]">
      <span className="w-6" style={{ color: req.color }}>{req.id}</span>
      <div className="flex-1 h-4 bg-dark-surface-elevated/30 rounded-sm overflow-hidden relative">
        {mode === 'contig' ? (
          <>
            {/* reserved full bar (hatched), used solid */}
            <div
              className="absolute inset-0 opacity-20 rounded-sm"
              style={{
                background: `repeating-linear-gradient(45deg, ${req.color}, ${req.color} 4px, transparent 4px, transparent 8px)`,
              }}
            />
            <div
              className="absolute top-0 bottom-0 left-0 rounded-sm"
              style={{ width: `${usedFrac * 100}%`, backgroundColor: req.color, opacity: 0.9 }}
            />
          </>
        ) : (
          <>
            <div
              className="absolute top-0 bottom-0 left-0"
              style={{
                width: `${pagedReservedFrac * 100}%`,
                backgroundColor: req.color,
                opacity: 0.25,
              }}
            />
            <div
              className="absolute top-0 bottom-0 left-0"
              style={{ width: `${usedFrac * 100}%`, backgroundColor: req.color, opacity: 0.9 }}
            />
          </>
        )}
      </div>
      <span className="w-14 text-right tabular-nums text-dark-text-muted">{req.used}/{MAX_SEQ_LEN}</span>
    </div>
  )
}

function MetricRow({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex justify-between gap-3 py-0.5">
      <span className="text-dark-text-disabled">{label}</span>
      <span className={cn('tabular-nums text-dark-text-primary', accent)}>{value}</span>
    </div>
  )
}
