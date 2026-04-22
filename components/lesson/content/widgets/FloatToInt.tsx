'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Draw a Gaussian float32 weight distribution. Overlay int8 quantization
// levels (256 buckets between the chosen min/max). The user drags a clip
// range slider and clicks any weight to see its int8 code and round-trip
// error. A dequantized histogram (reconstructed weights) appears below.

const N = 1000
const MU = 0
const SIGMA = 0.35

// Deterministic RNG + Gaussian sampler so SSR matches client.
function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gauss(rng: () => number): number {
  const u = Math.max(rng(), 1e-9)
  const v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

const WEIGHTS: number[] = (() => {
  const rng = mulberry32(7)
  const out: number[] = []
  for (let i = 0; i < N; i++) out.push(MU + SIGMA * gauss(rng))
  // Inject a few tail points so the distribution has visible tails
  out[10] = 1.4
  out[20] = -1.3
  out[30] = 1.1
  return out
})()

const BINS = 48

function histogram(vals: number[], lo: number, hi: number): number[] {
  const out = new Array(BINS).fill(0)
  const step = (hi - lo) / BINS
  for (const v of vals) {
    if (v < lo || v > hi) continue
    const idx = Math.min(BINS - 1, Math.floor((v - lo) / step))
    out[idx] += 1
  }
  return out
}

function quantize(w: number, clipMin: number, clipMax: number): {
  code: number
  dequant: number
  clipped: boolean
} {
  const range = clipMax - clipMin
  const scale = range / 255
  const clamped = Math.max(clipMin, Math.min(clipMax, w))
  const code = Math.round((clamped - clipMin) / scale) - 128 // [-128, 127]
  const dequant = (code + 128) * scale + clipMin
  return { code, dequant, clipped: w < clipMin || w > clipMax }
}

export default function FloatToInt() {
  const dataMin = -1.6
  const dataMax = 1.6
  const [clipMin, setClipMin] = useState(-1.0)
  const [clipMax, setClipMax] = useState(1.0)
  const [picked, setPicked] = useState<number | null>(42)

  const clipLo = Math.min(clipMin, clipMax)
  const clipHi = Math.max(clipMin, clipMax)

  const dequantized = useMemo(
    () => WEIGHTS.map((w) => quantize(w, clipLo, clipHi).dequant),
    [clipLo, clipHi],
  )

  const mse = useMemo(() => {
    let s = 0
    for (let i = 0; i < WEIGHTS.length; i++) {
      const d = WEIGHTS[i] - dequantized[i]
      s += d * d
    }
    return s / WEIGHTS.length
  }, [dequantized])

  const clippedCount = useMemo(
    () => WEIGHTS.filter((w) => w < clipLo || w > clipHi).length,
    [clipLo, clipHi],
  )

  const pickedW = picked !== null ? WEIGHTS[picked] : null
  const pickedQ = pickedW !== null ? quantize(pickedW, clipLo, clipHi) : null

  const topHist = histogram(WEIGHTS, dataMin, dataMax)
  const botHist = histogram(dequantized, dataMin, dataMax)
  const maxBin = Math.max(...topHist, ...botHist, 1)

  const scale = (clipHi - clipLo) / 255

  return (
    <WidgetFrame
      widgetName="FloatToInt"
      label="float32 weight → int8 code — drag clip range, click a bin to inspect"
      right={<span className="font-mono">N = {N} · 256 levels · scale = {scale.toFixed(5)}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="clip min"
            value={clipMin}
            min={-1.6}
            max={0}
            step={0.02}
            onChange={setClipMin}
            format={(v) => v.toFixed(2)}
            accent="accent-term-cyan"
          />
          <Slider
            label="clip max"
            value={clipMax}
            min={0}
            max={1.6}
            step={0.02}
            onChange={setClipMax}
            format={(v) => v.toFixed(2)}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="mse" value={mse.toExponential(2)} accent="text-term-rose" />
            <Readout label="clipped" value={String(clippedCount)} accent={clippedCount ? 'text-term-rose' : 'text-term-green'} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <div className="grid h-full grid-cols-1 md:grid-cols-[1fr_240px] gap-4 min-h-0">
          {/* Left: stacked histograms */}
          <div className="flex flex-col gap-2 min-h-0">
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
              float32 distribution · 256 int8 quantization ticks
            </div>
            <HistogramPane
              bins={topHist}
              maxBin={maxBin}
              dataMin={dataMin}
              dataMax={dataMax}
              clipLo={clipLo}
              clipHi={clipHi}
              color="#67e8f9"
              onPick={(binIdx) => {
                const lo = dataMin + (binIdx / BINS) * (dataMax - dataMin)
                const hi = dataMin + ((binIdx + 1) / BINS) * (dataMax - dataMin)
                const candidates = WEIGHTS
                  .map((w, i) => ({ w, i }))
                  .filter((x) => x.w >= lo && x.w < hi)
                if (candidates.length) {
                  const mid = candidates[Math.floor(candidates.length / 2)]
                  setPicked(mid.i)
                }
              }}
              pickedValue={pickedW ?? undefined}
              showTicks
            />
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
              dequantized distribution (round-trip) · note the discretization
            </div>
            <HistogramPane
              bins={botHist}
              maxBin={maxBin}
              dataMin={dataMin}
              dataMax={dataMax}
              clipLo={clipLo}
              clipHi={clipHi}
              color="#fbbf24"
            />
          </div>

          {/* Right: picked-weight inspector */}
          <div className="flex flex-col gap-2 min-w-0">
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
              picked weight
            </div>
            <div className="rounded border border-dark-border bg-dark-surface-elevated/40 p-3 font-mono text-[11px] flex flex-col gap-1.5">
              {pickedQ && pickedW !== null ? (
                <>
                  <InspectorRow label="w (fp32)" value={pickedW.toFixed(6)} accent="text-term-cyan" />
                  <InspectorRow
                    label="(w − min) / scale"
                    value={((Math.max(clipLo, Math.min(clipHi, pickedW)) - clipLo) / scale).toFixed(2)}
                  />
                  <InspectorRow
                    label="int8 code"
                    value={`${pickedQ.code}`}
                    accent="text-term-amber"
                  />
                  <InspectorRow
                    label="deq (fp32)"
                    value={pickedQ.dequant.toFixed(6)}
                    accent="text-term-amber"
                  />
                  <InspectorRow
                    label="|error|"
                    value={Math.abs(pickedW - pickedQ.dequant).toExponential(2)}
                    accent={pickedQ.clipped ? 'text-term-rose' : 'text-term-green'}
                  />
                  {pickedQ.clipped && (
                    <div className="mt-1 text-[10px] text-term-rose">
                      clipped: outside [{clipLo.toFixed(2)}, {clipHi.toFixed(2)}]
                    </div>
                  )}
                </>
              ) : (
                <div className="text-dark-text-muted">click a histogram bin</div>
              )}
            </div>

            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-1">
              formula
            </div>
            <div className="rounded border border-dark-border bg-dark-surface-elevated/40 p-3 font-mono text-[10.5px] leading-relaxed text-dark-text-muted">
              <div><span className="text-term-cyan">scale</span> = (max − min) / 255</div>
              <div><span className="text-term-amber">code</span> = round((w − min)/scale) − 128</div>
              <div><span className="text-term-amber">deq</span>  = (code + 128)·scale + min</div>
              <div className="mt-2 text-term-rose">values outside [min,max] are clipped</div>
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function InspectorRow({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-dark-text-disabled">{label}</span>
      <span className={cn('tabular-nums text-dark-text-primary', accent)}>{value}</span>
    </div>
  )
}

function HistogramPane({
  bins,
  maxBin,
  dataMin,
  dataMax,
  clipLo,
  clipHi,
  color,
  onPick,
  pickedValue,
  showTicks,
}: {
  bins: number[]
  maxBin: number
  dataMin: number
  dataMax: number
  clipLo: number
  clipHi: number
  color: string
  onPick?: (binIdx: number) => void
  pickedValue?: number
  showTicks?: boolean
}) {
  const W = 600
  const H = 110
  const padL = 28
  const padR = 8
  const padT = 6
  const padB = 18
  const plotW = W - padL - padR
  const plotH = H - padT - padB

  const toSx = (v: number) => padL + ((v - dataMin) / (dataMax - dataMin)) * plotW
  const binW = plotW / BINS

  return (
    <div className="flex-1 min-h-0 rounded bg-dark-bg/60 border border-dark-border/60">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="w-full h-full">
        {/* Clipped-out regions shaded */}
        <rect x={padL} y={padT} width={toSx(clipLo) - padL} height={plotH} fill="#fb7185" opacity={0.06} />
        <rect x={toSx(clipHi)} y={padT} width={padL + plotW - toSx(clipHi)} height={plotH} fill="#fb7185" opacity={0.06} />

        {/* Axis grid */}
        {[-1.5, -1, -0.5, 0, 0.5, 1, 1.5].map((x) => (
          <g key={x}>
            <line x1={toSx(x)} y1={padT} x2={toSx(x)} y2={padT + plotH} stroke="#1e1e1e" strokeWidth={1} />
            <text x={toSx(x)} y={H - 4} fontSize="8" textAnchor="middle" fill="#555" fontFamily="JetBrains Mono, monospace">
              {x}
            </text>
          </g>
        ))}

        {/* 256 int8 level ticks — subsampled to 32 visible */}
        {showTicks &&
          Array.from({ length: 33 }).map((_, i) => {
            const v = clipLo + (i / 32) * (clipHi - clipLo)
            return (
              <line
                key={i}
                x1={toSx(v)}
                y1={padT + plotH - 3}
                x2={toSx(v)}
                y2={padT + plotH}
                stroke={color}
                strokeWidth={1}
                opacity={0.55}
              />
            )
          })}

        {/* Clip range markers */}
        <line x1={toSx(clipLo)} y1={padT} x2={toSx(clipLo)} y2={padT + plotH} stroke="#67e8f9" strokeWidth={1.2} strokeDasharray="4 3" />
        <line x1={toSx(clipHi)} y1={padT} x2={toSx(clipHi)} y2={padT + plotH} stroke="#fbbf24" strokeWidth={1.2} strokeDasharray="4 3" />

        {/* Bars */}
        {bins.map((c, i) => {
          const v = dataMin + ((i + 0.5) / BINS) * (dataMax - dataMin)
          const isClipped = v < clipLo || v > clipHi
          const bh = (c / maxBin) * plotH
          return (
            <rect
              key={i}
              x={padL + i * binW + 0.5}
              y={padT + plotH - bh}
              width={Math.max(1, binW - 1)}
              height={bh}
              fill={isClipped ? '#fb7185' : color}
              opacity={isClipped ? 0.5 : 0.8}
              onClick={() => onPick?.(i)}
              className={onPick ? 'cursor-pointer' : undefined}
            />
          )
        })}

        {/* Picked-weight marker */}
        {pickedValue !== undefined && (
          <g>
            <line
              x1={toSx(pickedValue)}
              y1={padT}
              x2={toSx(pickedValue)}
              y2={padT + plotH}
              stroke="#f472b6"
              strokeWidth={1.5}
            />
            <circle cx={toSx(pickedValue)} cy={padT + 5} r={3} fill="#f472b6" />
          </g>
        )}
      </svg>
    </div>
  )
}
