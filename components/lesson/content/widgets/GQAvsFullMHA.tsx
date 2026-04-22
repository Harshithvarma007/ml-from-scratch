'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Side-by-side metric comparison between full MHA and GQA. User picks n_heads
// and n_kv_heads. We compute: KV cache bytes (per-token, per-layer), memory
// bandwidth fraction vs MHA, inference speedup, and a rough quality delta
// (empirical fit: MQA loses ~0.25 ppl, GQA ~0.05, MHA 0). Reference: Llama-2
// 70B uses 64 query heads + 8 KV heads — a 4× cache reduction.

const HEAD_DIM = 128
const BYTES_FP16 = 2

const HEAD_OPTIONS = [8, 16, 32]
const KV_FACTORS = [1, 2, 4, 8]

function cacheBytesPerTokenPerLayer(n_kv: number): number {
  return 2 * n_kv * HEAD_DIM * BYTES_FP16
}

// Rough quality delta in perplexity vs. full MHA. Fit mimics empirical GQA
// numbers: MQA (kv=1) worst, GQA in between, MHA zero.
function qualityDelta(n_heads: number, n_kv: number): number {
  if (n_kv === n_heads) return 0
  const frac = n_kv / n_heads
  // Monotone in frac: MQA (frac=1/n_heads) → 0.25ppl, GQA frac≥1/8 → ~0.05ppl
  if (n_kv === 1) return 0.25
  return 0.05 + (0.25 - 0.05) * (1 / (n_kv + 1))
}

export default function GQAvsFullMHA() {
  const [n_heads, setNHeads] = useState(32)
  const [kv_factor, setKvFactor] = useState(4) // kv = n_heads / kv_factor
  const n_kv = Math.max(1, Math.floor(n_heads / kv_factor))

  const metrics = useMemo(() => {
    const mhaBytes = cacheBytesPerTokenPerLayer(n_heads)
    const gqaBytes = cacheBytesPerTokenPerLayer(n_kv)
    // Memory bandwidth during decode is roughly proportional to cache size
    const memBw = gqaBytes / mhaBytes
    // Speedup at decode — cache-bandwidth-bound, so inverse of bw fraction
    const speedup = mhaBytes / Math.max(gqaBytes, 1)
    const qd = qualityDelta(n_heads, n_kv)

    return [
      {
        label: 'KV cache / token / layer',
        mha: mhaBytes,
        gqa: gqaBytes,
        fmt: (v: number) => `${v} B`,
        gqaWins: true,
        suffix: 'bytes',
      },
      {
        label: 'memory bandwidth at decode',
        mha: 1.0,
        gqa: memBw,
        fmt: (v: number) => `${v.toFixed(2)}×`,
        gqaWins: true,
        suffix: 'relative',
      },
      {
        label: 'decode speedup',
        mha: 1.0,
        gqa: speedup,
        fmt: (v: number) => `${v.toFixed(2)}×`,
        gqaWins: true,
        suffix: 'relative',
      },
      {
        label: 'quality regression (vs MHA)',
        mha: 0,
        gqa: qd,
        fmt: (v: number) => `+${v.toFixed(2)}`,
        gqaWins: false, // MHA wins here
        suffix: 'Δ ppl',
      },
    ]
  }, [n_heads, n_kv])

  const regime = n_kv === 1 ? 'MQA' : n_kv === n_heads ? 'MHA' : 'GQA'

  return (
    <WidgetFrame
      widgetName="GQAvsFullMHA"
      label="GQA vs full MHA — the four levers"
      right={<span className="font-mono">Llama-2 70B reference: 64 Q heads · 8 KV heads</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2 font-mono text-[11px]">
            <span className="text-dark-text-secondary">n_heads</span>
            <div className="flex items-center gap-0.5">
              {HEAD_OPTIONS.map((h) => (
                <button
                  key={h}
                  onClick={() => setNHeads(h)}
                  className={cn(
                    'px-2 py-0.5 rounded text-[10.5px] transition-all',
                    n_heads === h
                      ? 'bg-dark-accent text-white'
                      : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                  )}
                >
                  {h}
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2 font-mono text-[11px]">
            <span className="text-dark-text-secondary">n_kv</span>
            <div className="flex items-center gap-0.5">
              {KV_FACTORS.map((f) => {
                const v = Math.max(1, Math.floor(n_heads / f))
                return (
                  <button
                    key={f}
                    onClick={() => setKvFactor(f)}
                    className={cn(
                      'px-2 py-0.5 rounded text-[10.5px] transition-all',
                      kv_factor === f
                        ? 'bg-dark-accent text-white'
                        : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                    )}
                  >
                    {v}
                  </button>
                )
              })}
            </div>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="regime" value={regime} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[1fr_260px] gap-5 overflow-auto">
        <div className="flex flex-col gap-4 min-w-0">
          {metrics.map((m) => (
            <MetricCompare key={m.label} {...m} />
          ))}
        </div>

        <div className="flex flex-col gap-3 min-w-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            named configurations
          </div>
          <div className="flex flex-col gap-1 font-mono text-[10.5px]">
            <KnownCfg name="Llama-2 7B (MHA)" h={32} kv={32} active={n_heads === 32 && n_kv === 32} onClick={() => { setNHeads(32); setKvFactor(1) }} />
            <KnownCfg name="Llama-2 70B (GQA)" h={16} kv={2} active={n_heads === 16 && n_kv === 2} onClick={() => { setNHeads(16); setKvFactor(8) }} />
            <KnownCfg name="PaLM MQA (32→1)" h={32} kv={1} active={n_heads === 32 && n_kv === 1} onClick={() => { setNHeads(32); setKvFactor(32) }} />
            <KnownCfg name="Mistral-7B GQA" h={32} kv={8} active={n_heads === 32 && n_kv === 8} onClick={() => { setNHeads(32); setKvFactor(4) }} />
          </div>

          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-2">
            current config
          </div>
          <div className="rounded border border-dark-border bg-dark-surface-elevated/40 p-3 font-mono text-[10.5px] leading-relaxed text-dark-text-muted">
            <div className="flex justify-between">
              <span>n_heads</span>
              <span className="text-dark-text-primary tabular-nums">{n_heads}</span>
            </div>
            <div className="flex justify-between">
              <span>n_kv_heads</span>
              <span className="text-dark-text-primary tabular-nums">{n_kv}</span>
            </div>
            <div className="flex justify-between">
              <span>queries / KV</span>
              <span className="text-term-amber tabular-nums">{Math.floor(n_heads / n_kv)}</span>
            </div>
            <div className="flex justify-between">
              <span>regime</span>
              <span className="text-term-amber tabular-nums">{regime}</span>
            </div>
          </div>

          <div className="mt-1 font-mono text-[10.5px] leading-relaxed text-dark-text-muted">
            GQA keeps the quality of MHA for shockingly few KV heads — 8 is the sweet spot at scale. MQA is faster but measurable ppl regression on hard tasks.
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function MetricCompare({
  label,
  mha,
  gqa,
  fmt,
  gqaWins,
  suffix,
}: {
  label: string
  mha: number
  gqa: number
  fmt: (v: number) => string
  gqaWins: boolean
  suffix: string
}) {
  const max = Math.max(Math.abs(mha), Math.abs(gqa), 1e-6)
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between font-mono text-[11px]">
        <span className="text-dark-text-secondary uppercase tracking-wider text-[10px]">{label}</span>
        <span className="text-dark-text-disabled text-[10px]">{suffix}</span>
      </div>
      <div className="flex items-center gap-3 font-mono text-[11px]">
        <span className="w-10 text-term-cyan">MHA</span>
        <div className="flex-1 h-5 bg-dark-surface-elevated/40 rounded overflow-hidden">
          <div
            className={cn('h-full rounded', gqaWins ? 'bg-term-cyan/55' : 'bg-term-cyan/85')}
            style={{ width: `${(Math.abs(mha) / max) * 100}%` }}
          />
        </div>
        <span className={cn('w-24 text-right tabular-nums', gqaWins ? 'text-dark-text-primary' : 'text-term-cyan')}>
          {fmt(mha)}
        </span>
      </div>
      <div className="flex items-center gap-3 font-mono text-[11px]">
        <span className="w-10 text-term-amber">GQA</span>
        <div className="flex-1 h-5 bg-dark-surface-elevated/40 rounded overflow-hidden">
          <div
            className={cn('h-full rounded', gqaWins ? 'bg-term-amber/85' : 'bg-term-amber/55')}
            style={{ width: `${(Math.abs(gqa) / max) * 100}%` }}
          />
        </div>
        <span className={cn('w-24 text-right tabular-nums', gqaWins ? 'text-term-amber' : 'text-dark-text-primary')}>
          {fmt(gqa)}
        </span>
      </div>
    </div>
  )
}

function KnownCfg({ name, h, kv, active, onClick }: { name: string; h: number; kv: number; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'flex justify-between items-center px-2 py-1 rounded transition-all border',
        active
          ? 'border-term-amber text-term-amber bg-term-amber/10'
          : 'border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
      )}
    >
      <span>{name}</span>
      <span className="tabular-nums text-dark-text-disabled">{h}Q / {kv}KV</span>
    </button>
  )
}
