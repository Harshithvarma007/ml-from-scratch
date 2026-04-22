'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// For a tensor of shape (B, S, D), count the elementary ops each normalisation
// performs. LayerNorm does an extra mean + subtraction that RMS skips. We
// translate ops into a projected millisecond cost (illustrative — not a real
// benchmark) so the reader can feel the "~10-15% faster" claim.

export default function NormOpCost() {
  const [B, setB] = useState(8)
  const [S, setS] = useState(2048)
  const [D, setD] = useState(4096)

  const counts = useMemo(() => {
    const N = B * S // number of "positions" each norm is applied to
    // Per position:
    // LayerNorm: sum for mean (D), subtract+square (D), sum for var (D),
    //   rsqrt+divide (D), then gamma/beta scale (D). ~5D ops.
    // RMSNorm: square (D), sum (D), rsqrt+divide (D), gamma (D). ~4D ops.
    // BatchNorm: similar to LN but across batch axis — same op count per element.
    const lnOps = N * 5 * D
    const rmsOps = N * 4 * D
    // Pretend 1 ns per op on an H100-class FLOP budget (illustrative only).
    const lnMs = lnOps / 1e9
    const rmsMs = rmsOps / 1e9
    return { N, lnOps, rmsOps, lnMs, rmsMs, saved: (lnOps - rmsOps) / lnOps }
  }, [B, S, D])

  return (
    <WidgetFrame
      widgetName="NormOpCost"
      label="the compute tradeoff — why large LLMs pick RMSNorm"
      right={
        <>
          <span className="font-mono">ops counted per position · ε and affine folded in</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="B (batch)"
            value={B}
            min={1}
            max={64}
            step={1}
            onChange={(v) => setB(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-cyan"
          />
          <Slider
            label="S (seq len)"
            value={S}
            min={128}
            max={16384}
            step={64}
            onChange={(v) => setS(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-dark-accent"
          />
          <Slider
            label="D (features)"
            value={D}
            min={128}
            max={12288}
            step={64}
            onChange={(v) => setD(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-purple"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="RMS savings"
              value={`${(counts.saved * 100).toFixed(0)}%`}
              accent="text-term-amber"
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 overflow-auto">
        <div className="max-w-[820px] mx-auto">
          <div className="text-[10.5px] font-mono text-dark-text-disabled mb-2">
            tensor shape ({B}, {S}, {D}) → {fmt(counts.N)} normalization positions
          </div>

          {/* Op comparison bars */}
          <div className="space-y-4 mb-5">
            <CostBar
              name="LayerNorm"
              ops={counts.lnOps}
              maxOps={counts.lnOps}
              color="#a78bfa"
              breakdown={[
                { label: 'mean', cost: 1 },
                { label: 'subtract', cost: 1 },
                { label: 'var', cost: 1 },
                { label: 'rsqrt + div', cost: 1 },
                { label: 'γ · + β', cost: 1 },
              ]}
            />
            <CostBar
              name="RMSNorm"
              ops={counts.rmsOps}
              maxOps={counts.lnOps}
              color="#fbbf24"
              breakdown={[
                { label: 'square', cost: 1 },
                { label: 'mean', cost: 1 },
                { label: 'rsqrt + div', cost: 1 },
                { label: 'γ ·', cost: 1 },
              ]}
            />
          </div>

          {/* Sub-op breakdown table */}
          <div className="rounded border border-dark-border overflow-hidden">
            <table className="w-full font-mono text-[11px]">
              <thead className="bg-dark-surface-elevated/40 text-dark-text-disabled uppercase text-[10px]">
                <tr>
                  <th className="text-left px-3 py-2">sub-op</th>
                  <th className="text-center px-3 py-2">LayerNorm</th>
                  <th className="text-center px-3 py-2">RMSNorm</th>
                </tr>
              </thead>
              <tbody>
                <OpRow sub="compute mean" ln="✓" rms="—" savings />
                <OpRow sub="subtract mean" ln="✓" rms="—" savings />
                <OpRow sub="compute mean(x²)" ln="via var" rms="✓" />
                <OpRow sub="compute var" ln="✓" rms="— (uses mean(x²))" />
                <OpRow sub="rsqrt + divide" ln="✓" rms="✓" />
                <OpRow sub="γ · scale" ln="✓" rms="✓" />
                <OpRow sub="+ β shift" ln="✓" rms="— (no β)" savings />
              </tbody>
            </table>
          </div>

          <div className="mt-4 p-3 rounded border border-dark-border bg-dark-surface-elevated/30 text-[11.5px] font-sans text-dark-text-secondary leading-relaxed">
            <strong className="text-dark-text-primary">The real savings are memory and
            latency.</strong>{' '}
            Each norm op is memory-bound — the arithmetic is cheap, the slow part is reading
            and writing the tensor. Skipping mean computation cuts one full read of the
            activation tensor. On a modern GPU with memory bandwidth of ~3 TB/s, a single
            LayerNorm over a{' '}
            <code className="text-dark-text-primary">(8, 2048, 4096)</code> tensor takes
            about ~200 µs; RMSNorm comes in around ~170 µs. A 15% per-norm saving, summed
            over every transformer block, becomes a noticeable speedup at scale.
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function CostBar({
  name,
  ops,
  maxOps,
  color,
  breakdown,
}: {
  name: string
  ops: number
  maxOps: number
  color: string
  breakdown: Array<{ label: string; cost: number }>
}) {
  const total = breakdown.reduce((s, b) => s + b.cost, 0)
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <span className="font-mono text-[11px] uppercase tracking-wider" style={{ color }}>
          {name}
        </span>
        <span className="font-mono text-[11px] text-dark-text-muted">
          {fmt(ops)} ops
        </span>
      </div>
      <div className="h-8 flex rounded overflow-hidden bg-dark-bg border border-dark-border">
        {breakdown.map((b, i) => (
          <div
            key={i}
            className="flex items-center justify-center text-[9.5px] font-mono text-dark-bg overflow-hidden whitespace-nowrap"
            style={{
              width: `${(ops / maxOps) * (b.cost / total) * 100}%`,
              backgroundColor: color + (i % 2 === 0 ? 'cc' : '99'),
            }}
          >
            {b.label}
          </div>
        ))}
      </div>
    </div>
  )
}

function OpRow({
  sub,
  ln,
  rms,
  savings,
}: {
  sub: string
  ln: string
  rms: string
  savings?: boolean
}) {
  return (
    <tr className={cn('border-t border-dark-border', savings && 'bg-term-amber/5')}>
      <td className="px-3 py-1.5 text-dark-text-primary">{sub}</td>
      <td
        className={cn(
          'px-3 py-1.5 text-center',
          ln === '—' ? 'text-dark-text-disabled' : 'text-term-purple'
        )}
      >
        {ln}
      </td>
      <td
        className={cn(
          'px-3 py-1.5 text-center',
          rms.startsWith('—') ? 'text-term-amber' : rms === '✓' ? 'text-term-amber' : 'text-dark-text-muted'
        )}
      >
        {rms}
      </td>
    </tr>
  )
}

function fmt(n: number): string {
  if (n < 1000) return String(Math.round(n))
  if (n < 1e6) return (n / 1e3).toFixed(1) + 'K'
  if (n < 1e9) return (n / 1e6).toFixed(1) + 'M'
  if (n < 1e12) return (n / 1e9).toFixed(1) + 'B'
  return (n / 1e12).toFixed(1) + 'T'
}
