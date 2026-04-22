'use client'

import { useEffect, useMemo, useRef } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Throughput vs. number of concurrent requests for three batching modes:
//   static:     flat — saturates at max_batch and wastes on stragglers
//   dynamic:    scales in windows — better than static but quantized
//   continuous: near-linear — each freed slot refills instantly
// Then a second panel: p50 / p99 latency bars for each mode.

const MAX_CONCURRENCY = 32

function staticThroughput(n: number): number {
  const maxBatch = 8
  if (n <= 1) return 40
  // once you hit max_batch you get the batch speedup, but wasted tails
  // bring the effective throughput below ideal. Flat-ish beyond 16.
  const effective = Math.min(n, maxBatch)
  return 40 + effective * 28 * (1 - 0.15) // 15% waste on stragglers
}

function dynamicThroughput(n: number): number {
  // Windowed refill: scales decently but plateaus around 2 batches worth
  const maxBatch = 16
  const effective = Math.min(n, maxBatch)
  return 40 + effective * 34
}

function continuousThroughput(n: number): number {
  // Near-linear up to ~24 where memory contention kicks in, then sub-linear
  if (n <= 24) return 40 + n * 40
  return 40 + 24 * 40 + (n - 24) * 12
}

const MODES = [
  { key: 'static', label: 'static', color: '#fb7185', fn: staticThroughput },
  { key: 'dynamic', label: 'dynamic', color: '#fbbf24', fn: dynamicThroughput },
  { key: 'continuous', label: 'continuous', color: '#4ade80', fn: continuousThroughput },
] as const

const P50_LATENCY = { static: 180, dynamic: 140, continuous: 95 } as const
const P99_LATENCY = { static: 620, dynamic: 380, continuous: 180 } as const

export default function ThroughputComparison() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const series = useMemo(() => {
    return MODES.map((m) => ({
      ...m,
      data: Array.from({ length: MAX_CONCURRENCY }, (_, i) => ({ n: i + 1, tps: m.fn(i + 1) })),
    }))
  }, [])

  const peak = useMemo(() => {
    const cont = series.find((s) => s.key === 'continuous')!
    const stat = series.find((s) => s.key === 'static')!
    const peakCont = Math.max(...cont.data.map((d) => d.tps))
    const peakStat = Math.max(...stat.data.map((d) => d.tps))
    return { cont: peakCont, stat: peakStat, ratio: peakCont / peakStat }
  }, [series])

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
      const yMax = Math.ceil(peak.cont / 100) * 100 + 50
      const yMin = 0
      const toSx = (n: number) => padL + ((n - xMin) / (xMax - xMin)) * plotW
      const toSy = (v: number) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH

      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.strokeStyle = '#1e1e1e'

      // grid
      for (let v = 0; v <= yMax; v += 200) {
        ctx.beginPath()
        ctx.moveTo(padL, toSy(v))
        ctx.lineTo(padL + plotW, toSy(v))
        ctx.stroke()
        ctx.textAlign = 'right'
        ctx.fillText(String(v), padL - 6, toSy(v) + 3)
      }
      for (let n = 1; n <= xMax; n += 4) {
        ctx.textAlign = 'center'
        ctx.fillText(String(n), toSx(n), padT + plotH + 14)
      }
      ctx.fillText('concurrent requests', padL + plotW / 2, padT + plotH + 28)
      ctx.save()
      ctx.translate(14, padT + plotH / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.textAlign = 'center'
      ctx.fillText('tokens/sec', 0, 0)
      ctx.restore()

      // lines
      series.forEach((s) => {
        ctx.strokeStyle = s.color
        ctx.lineWidth = 2.2
        ctx.beginPath()
        s.data.forEach((p, i) => {
          const x = toSx(p.n)
          const y = toSy(p.tps)
          if (i === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        })
        ctx.stroke()
      })

      // legend
      let lx = padL + 10
      const ly = padT + 6
      series.forEach((s) => {
        ctx.strokeStyle = s.color
        ctx.lineWidth = 2.2
        ctx.beginPath()
        ctx.moveTo(lx, ly + 8)
        ctx.lineTo(lx + 16, ly + 8)
        ctx.stroke()
        ctx.fillStyle = s.color
        ctx.textAlign = 'left'
        ctx.fillText(s.label, lx + 20, ly + 11)
        lx += 24 + ctx.measureText(s.label).width + 10
      })
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [series, peak])

  return (
    <WidgetFrame
      widgetName="ThroughputComparison"
      label="throughput & latency — three batching modes"
      right={<span className="font-mono">continuous ≈ {peak.ratio.toFixed(1)}× the peak of static</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-3">
            {MODES.map((m) => (
              <div key={m.key} className="flex items-center gap-1.5 font-mono text-[11px]">
                <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: m.color }} />
                <span style={{ color: m.color }}>{m.label}</span>
              </div>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="peak continuous" value={`${Math.round(peak.cont)} tok/s`} accent="text-term-green" />
            <Readout label="peak static" value={`${Math.round(peak.stat)} tok/s`} accent="text-term-rose" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <div className="grid h-full grid-cols-1 md:grid-cols-[1fr_260px] gap-4 min-h-0">
          {/* Line chart */}
          <div ref={boxRef} className="min-h-0 rounded border border-dark-border bg-dark-bg/60">
            <canvas ref={canvasRef} className="w-full h-full block" />
          </div>

          {/* Latency bars */}
          <div className="flex flex-col gap-3 min-w-0">
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
              latency (p50 vs. p99, ms)
            </div>
            <div className="rounded border border-dark-border bg-dark-surface-elevated/40 p-3 flex flex-col gap-3">
              {MODES.map((m) => (
                <LatencyCompare
                  key={m.key}
                  label={m.label}
                  color={m.color}
                  p50={P50_LATENCY[m.key]}
                  p99={P99_LATENCY[m.key]}
                  max={Math.max(...Object.values(P99_LATENCY))}
                />
              ))}
            </div>

            <div className="rounded border border-dark-border bg-dark-surface-elevated/40 p-3 font-mono text-[10.5px] leading-relaxed text-dark-text-muted">
              static batching waits for the longest sequence in every micro-batch — p99 balloons.
              continuous batching releases lanes the moment a request finishes, so short requests never stand in line behind long ones.
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function LatencyCompare({
  label,
  color,
  p50,
  p99,
  max,
}: {
  label: string
  color: string
  p50: number
  p99: number
  max: number
}) {
  return (
    <div className="flex flex-col gap-1 font-mono text-[10.5px]">
      <div className="flex justify-between">
        <span style={{ color }}>{label}</span>
        <span className="text-dark-text-disabled tabular-nums">
          p50 {p50}ms · p99 {p99}ms
        </span>
      </div>
      <div className="relative h-3 bg-dark-surface-elevated/60 rounded-sm overflow-hidden">
        <div
          className="absolute top-0 bottom-0"
          style={{
            width: `${(p99 / max) * 100}%`,
            backgroundColor: color,
            opacity: 0.3,
          }}
        />
        <div
          className="absolute top-0 bottom-0"
          style={{
            width: `${(p50 / max) * 100}%`,
            backgroundColor: color,
            opacity: 0.85,
          }}
        />
      </div>
      <div className={cn('flex justify-between text-[9px] text-dark-text-disabled')}>
        <span>p50</span>
        <span>p99</span>
      </div>
    </div>
  )
}
