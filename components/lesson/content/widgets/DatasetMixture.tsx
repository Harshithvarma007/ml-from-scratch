'use client'

// Four dataset sources with adjustable mixing weights. Sliders auto-normalize
// so they always sum to 100%. Top: horizontal stacked-bar of the mix. Bottom:
// scatter of 200 sampled "training examples" colored by source — positions
// stable across renders so you can see the mix shift. Readout shows effective
// tokens/epoch after mixing.

import { useMemo, useRef, useEffect, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'

type Source = {
  key: 'cc' | 'wiki' | 'books' | 'gh'
  name: string
  color: string
  tokens: number // in billions, roughly representative
}

const SOURCES: Source[] = [
  { key: 'cc',    name: 'CommonCrawl', color: '#fbbf24', tokens: 450 },
  { key: 'wiki',  name: 'Wikipedia',   color: '#67e8f9', tokens: 20 },
  { key: 'books', name: 'Books',       color: '#a78bfa', tokens: 50 },
  { key: 'gh',    name: 'GitHub',      color: '#4ade80', tokens: 80 },
]

const DEFAULTS = { cc: 0.60, wiki: 0.15, books: 0.15, gh: 0.10 }

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const N_SAMPLES = 200

// Precompute 200 stable (x, y) positions. Source assignment is decided per
// render based on current weights.
const POSITIONS = (() => {
  const rng = mulberry32(13)
  const arr: { x: number; y: number; u: number }[] = []
  for (let i = 0; i < N_SAMPLES; i++) {
    // slight clustering by source: use deterministic sample index
    arr.push({ x: rng(), y: rng(), u: rng() })
  }
  return arr
})()

export default function DatasetMixture() {
  const [weights, setWeights] = useState<Record<string, number>>(DEFAULTS)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const sum = weights.cc + weights.wiki + weights.books + weights.gh
  const norm = sum <= 0 ? DEFAULTS : Object.fromEntries(
    Object.entries(weights).map(([k, v]) => [k, v / sum])
  )

  // Assign each stable sample to a source by cumulative weight
  const assigned = useMemo(() => {
    const cuts = [
      { key: 'cc' as const, end: norm.cc },
      { key: 'wiki' as const, end: norm.cc + norm.wiki },
      { key: 'books' as const, end: norm.cc + norm.wiki + norm.books },
      { key: 'gh' as const, end: 1 },
    ]
    return POSITIONS.map(({ u, x, y }) => {
      for (const c of cuts) {
        if (u < c.end) return { key: c.key, x, y }
      }
      return { key: 'gh' as const, x, y }
    })
  }, [norm])

  // Draw scatter
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
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, w, h)

      const pad = 8
      const pw = w - pad * 2
      const ph = h - pad * 2
      // dot grid
      ctx.fillStyle = '#1a1a24'
      for (let gx = 0; gx < 12; gx++) {
        for (let gy = 0; gy < 8; gy++) {
          const x = pad + (gx / 11) * pw
          const y = pad + (gy / 7) * ph
          ctx.beginPath()
          ctx.arc(x, y, 1, 0, Math.PI * 2)
          ctx.fill()
        }
      }

      for (const a of assigned) {
        const x = pad + a.x * pw
        const y = pad + a.y * ph
        const color = SOURCES.find((s) => s.key === a.key)!.color
        ctx.fillStyle = color
        ctx.beginPath()
        ctx.arc(x, y, 3.5, 0, Math.PI * 2)
        ctx.fill()
        ctx.strokeStyle = color + '55'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.arc(x, y, 5.5, 0, Math.PI * 2)
        ctx.stroke()
      }
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [assigned])

  // effective tokens/epoch = sum(weight * tokens)
  const effTokens = SOURCES.reduce((a, s) => a + norm[s.key] * s.tokens, 0)

  // sample counts by source
  const counts = SOURCES.map((s) => assigned.filter((a) => a.key === s.key).length)

  return (
    <WidgetFrame
      widgetName="DatasetMixture"
      label="dataset mixture — 4 sources, auto-normalized weights"
      right={<span className="font-mono">{N_SAMPLES} sampled examples · normalize → 100%</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="eff. tok/epoch" value={`${effTokens.toFixed(1)} B`} accent="text-term-green" />
            <Readout label="sum" value={`${(sum * 100).toFixed(0)}% (auto-normalized)`} accent="text-dark-text-muted" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[280px_1fr] gap-4 overflow-hidden">
        {/* Left: sliders */}
        <div className="flex flex-col gap-3 min-h-0 overflow-hidden">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            mixing weights
          </div>
          {SOURCES.map((s, i) => {
            const raw = weights[s.key]
            const normalized = norm[s.key]
            return (
              <div key={s.key} className="flex flex-col gap-1">
                <div className="flex items-center justify-between font-mono text-[11px]">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: s.color }} />
                    <span className="text-dark-text-secondary">{s.name}</span>
                    <span className="text-dark-text-disabled">· {s.tokens} B tok</span>
                  </span>
                  <span className="tabular-nums" style={{ color: s.color }}>
                    {(normalized * 100).toFixed(1)}%
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={raw}
                  onChange={(e) =>
                    setWeights({ ...weights, [s.key]: Number(e.target.value) })
                  }
                  className="w-full h-1 rounded-full bg-dark-border cursor-pointer"
                  style={{ accentColor: s.color }}
                />
                <div className="text-[9.5px] font-mono text-dark-text-disabled tabular-nums">
                  {counts[i]} / {N_SAMPLES} examples
                </div>
              </div>
            )
          })}
        </div>

        {/* Right: stacked bar + scatter */}
        <div className="flex flex-col gap-3 min-h-0 overflow-hidden">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            mixture bar
          </div>
          <div className="relative h-7 rounded border border-dark-border overflow-hidden flex">
            {SOURCES.map((s) => {
              const pct = norm[s.key] * 100
              return (
                <div
                  key={s.key}
                  className="h-full flex items-center justify-center font-mono text-[10px] text-dark-bg"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: s.color,
                    opacity: pct < 3 ? 0.4 : 0.9,
                  }}
                >
                  {pct >= 8 ? `${pct.toFixed(0)}%` : ''}
                </div>
              )
            })}
          </div>
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            200 sampled training examples · colored by source
          </div>
          <div ref={boxRef} className="relative flex-1 min-h-0 border border-dark-border rounded bg-dark-bg overflow-hidden">
            <canvas ref={canvasRef} className="w-full h-full block" />
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            {SOURCES.map((s, i) => (
              <div key={s.key} className="flex items-center gap-1.5 font-mono text-[10px]">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: s.color }} />
                <span className="text-dark-text-secondary">{s.name}</span>
                <span className="text-dark-text-disabled tabular-nums">· {counts[i]}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

