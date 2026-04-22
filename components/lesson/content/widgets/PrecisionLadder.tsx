'use client'

import { useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Side-by-side comparison of fp32, fp16, bf16, int8, int4. For each: bit
// breakdown, dynamic range, memory per 1B params, plus a round-trip of a
// user-picked sample value. The bit breakdown bar shows sign / exponent /
// mantissa (or full-int widths) with distinct accents.

type Precision = {
  key: string
  name: string
  bits: number
  sign: number
  exp: number
  mantissa: number
  range: string
  color: string
  isInt?: boolean
  intRange?: [number, number]
}

const PRECISIONS: Precision[] = [
  { key: 'fp32', name: 'fp32',  bits: 32, sign: 1, exp: 8,  mantissa: 23, range: '±3.4e38',  color: '#67e8f9' },
  { key: 'fp16', name: 'fp16',  bits: 16, sign: 1, exp: 5,  mantissa: 10, range: '±6.5e4',   color: '#fbbf24' },
  { key: 'bf16', name: 'bf16',  bits: 16, sign: 1, exp: 8,  mantissa: 7,  range: '±3.4e38',  color: '#a78bfa' },
  { key: 'int8', name: 'int8',  bits: 8,  sign: 0, exp: 0,  mantissa: 8,  range: '−128…127', color: '#4ade80', isInt: true, intRange: [-128, 127] },
  { key: 'int4', name: 'int4',  bits: 4,  sign: 0, exp: 0,  mantissa: 4,  range: '−8…7',     color: '#fb7185', isInt: true, intRange: [-8, 7] },
]

// Simulate a round-trip quantization from the sample value `w`. For floats
// we bucket to the nearest representable mantissa step at the exponent of
// `w`. For ints we assume a per-tensor scale over a [−1, 1] range (typical
// for weight tensors that have been normalized).
function roundTrip(w: number, p: Precision): number {
  if (p.isInt && p.intRange) {
    const [lo, hi] = p.intRange
    const scale = 1 / hi // weights normalized to [-1, 1]
    const code = Math.max(lo, Math.min(hi, Math.round(w / scale)))
    return code * scale
  }
  if (w === 0) return 0
  const absW = Math.abs(w)
  // biased exponent ~= floor(log2(|w|))
  const e = Math.floor(Math.log2(absW))
  const step = Math.pow(2, e - p.mantissa)
  const q = Math.round(absW / step) * step
  return Math.sign(w) * q
}

function bytesPer1B(bits: number): string {
  const bytes = (1_000_000_000 * bits) / 8
  const gb = bytes / 1_000_000_000
  return `${gb.toFixed(2)} GB`
}

export default function PrecisionLadder() {
  const [sample, setSample] = useState(0.1234567)

  return (
    <WidgetFrame
      widgetName="PrecisionLadder"
      label="precision ladder — same weight, five storage formats"
      right={<span className="font-mono">1B weights · bit breakdown · round-trip error</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="sample w"
            value={sample}
            min={-1}
            max={1}
            step={0.0001}
            onChange={setSample}
            format={(v) => v.toFixed(6)}
            accent="accent-term-cyan"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="fp32 → int4 ratio" value="8×" accent="text-term-green" />
            <Readout label="int4 savings" value="87.5%" accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <div className="grid h-full grid-cols-[110px_1fr_120px_140px_140px] gap-x-3 gap-y-2 font-mono text-[11px] items-center auto-rows-min">
          <HeaderCell>format</HeaderCell>
          <HeaderCell>bit layout</HeaderCell>
          <HeaderCell>range</HeaderCell>
          <HeaderCell>mem / 1B</HeaderCell>
          <HeaderCell>round-trip</HeaderCell>

          {PRECISIONS.map((p) => {
            const deq = roundTrip(sample, p)
            const err = Math.abs(sample - deq)
            return (
              <FormatRow
                key={p.key}
                p={p}
                sample={sample}
                dequant={deq}
                err={err}
              />
            )
          })}
        </div>

        <div className="mt-4 rounded border border-dark-border bg-dark-surface-elevated/40 p-3 font-mono text-[10.5px] leading-relaxed text-dark-text-muted">
          <div className="text-term-cyan mb-1">fp16 vs. bf16 — same 16 bits, very different trade-off</div>
          fp16 keeps 10 mantissa bits + 5 exponent — high precision in a narrow range; overflows at 6.5e4.
          bf16 keeps 7 mantissa + 8 exponent — same huge dynamic range as fp32 with halved memory. Modern training uses bf16.
        </div>
      </div>
    </WidgetFrame>
  )
}

function HeaderCell({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[9.5px] uppercase tracking-wider text-dark-text-disabled pb-1 border-b border-dark-border">
      {children}
    </div>
  )
}

function FormatRow({
  p,
  sample,
  dequant,
  err,
}: {
  p: Precision
  sample: number
  dequant: number
  err: number
}) {
  return (
    <>
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-sm" style={{ backgroundColor: p.color }} />
        <span className="text-dark-text-primary font-bold" style={{ color: p.color }}>{p.name}</span>
      </div>

      <BitLayout p={p} />

      <div className="tabular-nums text-dark-text-secondary">{p.range}</div>

      <div className="flex items-center gap-2">
        <div className="flex-1 h-2 bg-dark-surface-elevated/40 rounded-sm overflow-hidden">
          <div
            className="h-full rounded-sm"
            style={{
              width: `${(p.bits / 32) * 100}%`,
              backgroundColor: p.color,
              opacity: 0.75,
            }}
          />
        </div>
        <span className="tabular-nums text-dark-text-primary">{bytesPer1B(p.bits)}</span>
      </div>

      <div className="flex flex-col gap-0.5 tabular-nums">
        <span style={{ color: p.color }}>{dequant.toFixed(6)}</span>
        <span className={cn('text-[10px]', err > 0.01 ? 'text-term-rose' : 'text-dark-text-disabled')}>
          |e| = {err.toExponential(2)}
        </span>
      </div>
    </>
  )
}

function BitLayout({ p }: { p: Precision }) {
  // Show 32 boxes, fill the first `p.bits` with section colors.
  const cells: { kind: 's' | 'e' | 'm' | 'int' | 'empty'; color: string }[] = []
  for (let i = 0; i < p.bits; i++) {
    if (p.isInt) {
      cells.push({ kind: 'int', color: p.color })
    } else if (i < p.sign) {
      cells.push({ kind: 's', color: '#f472b6' })
    } else if (i < p.sign + p.exp) {
      cells.push({ kind: 'e', color: '#a78bfa' })
    } else {
      cells.push({ kind: 'm', color: '#67e8f9' })
    }
  }
  while (cells.length < 32) cells.push({ kind: 'empty', color: '#1e1e1e' })

  const legend = p.isInt
    ? `${p.bits}-bit signed integer`
    : `1 sign · ${p.exp} exp · ${p.mantissa} mantissa`

  return (
    <div className="flex flex-col gap-1">
      <div className="flex gap-[1px] h-4">
        {cells.map((c, i) => (
          <div
            key={i}
            className={cn(
              'flex-1 rounded-[1px]',
              c.kind === 'empty' && 'border border-dashed border-dark-border opacity-40',
            )}
            style={c.kind !== 'empty' ? { backgroundColor: c.color, opacity: 0.8 } : undefined}
          />
        ))}
      </div>
      <div className="text-[9.5px] text-dark-text-muted">{legend}</div>
    </div>
  )
}
