'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Stacked-bar view of where parameters live in a GPT: embeddings / attention
// QKV / attention out / MLP up / MLP down / norms / lm_head. d_model, n_layers
// and vocab_size are presets — they scale each component and the highlighted
// dominant component shifts as scale changes.

const D_PRESETS = [256, 512, 1024, 2048]
const L_PRESETS = [6, 12, 24, 48]
const V_PRESETS = [16384, 32768, 50257]

type Bucket = {
  key: string
  name: string
  color: string
  compute: (d: number, L: number, V: number, ff: number) => number
}

const BUCKETS: Bucket[] = [
  {
    key: 'embed',
    name: 'token + pos embedding',
    color: '#67e8f9',
    compute: (d, _L, V) => V * d + 1024 * d,
  },
  {
    key: 'qkv',
    name: 'attention Q/K/V',
    color: '#a78bfa',
    compute: (d, L) => L * (3 * d * d + 3 * d),
  },
  {
    key: 'attn_out',
    name: 'attention out proj',
    color: '#fbbf24',
    compute: (d, L) => L * (d * d + d),
  },
  {
    key: 'mlp_up',
    name: 'MLP up (d → 4d)',
    color: '#f472b6',
    compute: (d, L, _V, ff) => L * (d * ff + ff),
  },
  {
    key: 'mlp_down',
    name: 'MLP down (4d → d)',
    color: '#4ade80',
    compute: (d, L, _V, ff) => L * (ff * d + d),
  },
  {
    key: 'norms',
    name: 'layer norms',
    color: '#f87171',
    compute: (d, L) => L * 4 * d + 2 * d,
  },
  {
    key: 'lm_head',
    name: 'lm_head (tied)',
    color: '#fb7185',
    compute: () => 0, // tied to embedding, no net params
  },
]

function formatBig(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`
  return String(n)
}

export default function ParameterBreakdown() {
  const [d, setD] = useState(768)
  const [L, setL] = useState(12)
  const [V, setV] = useState(50257)

  const counts = useMemo(() => {
    const ff = 4 * d
    return BUCKETS.map((b) => ({ ...b, count: b.compute(d, L, V, ff) }))
  }, [d, L, V])

  const total = counts.reduce((a, b) => a + b.count, 0) || 1
  const dominant = counts.reduce((a, b) => (b.count > a.count ? b : a), counts[0])

  const presetName =
    d === 768 && L === 12 && V === 50257 ? 'GPT-2 small' :
    d === 1024 && L === 24 && V === 50257 ? 'GPT-2 medium' :
    d === 2048 && L === 48 && V === 50257 ? 'GPT-2 XL–ish' :
    d === 256 && L === 6 ? 'toy' :
    'custom'

  return (
    <WidgetFrame
      widgetName="ParameterBreakdown"
      label="where the parameters live — per-bucket breakdown"
      right={<span className="font-mono">{presetName} · d={d} · L={L} · V={formatBig(V)}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <PresetRow label="d_model" options={D_PRESETS} value={d} onChange={setD} fmt={String} />
          <PresetRow label="n_layers" options={L_PRESETS} value={L} onChange={setL} fmt={String} />
          <PresetRow label="vocab" options={V_PRESETS} value={V} onChange={setV} fmt={(v) => (v >= 1024 ? `${v / 1024}k` : String(v))} />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="total" value={formatBig(total)} accent="text-term-amber" />
            <Readout label="dominant" value={dominant.name} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[1fr_280px] gap-5 overflow-hidden">
        {/* Left: stacked bar + breakdown list */}
        <div className="flex flex-col gap-3 min-w-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            total parameter stack (click a preset to re-scale)
          </div>
          <div className="relative h-10 bg-dark-surface-elevated/40 rounded overflow-hidden flex">
            {counts.map((c) => {
              const pct = (c.count / total) * 100
              if (pct < 0.05) return null
              return (
                <div
                  key={c.key}
                  className="h-full flex items-center justify-center relative group"
                  style={{ width: `${pct}%`, backgroundColor: c.color, opacity: 0.8 }}
                  title={`${c.name}: ${formatBig(c.count)} (${pct.toFixed(1)}%)`}
                >
                  {pct > 6 && (
                    <span className="text-[9.5px] font-mono text-dark-bg font-bold">
                      {pct.toFixed(0)}%
                    </span>
                  )}
                </div>
              )
            })}
          </div>

          <div className="flex flex-col gap-1.5 mt-1">
            {counts.map((c) => {
              const pct = (c.count / total) * 100
              const isDom = c.key === dominant.key
              return (
                <div key={c.key} className="flex items-center gap-3 font-mono text-[11px]">
                  <span className="w-3 h-3 rounded-sm shrink-0" style={{ backgroundColor: c.color }} />
                  <span className={cn('w-44 truncate', isDom ? 'text-term-amber' : 'text-dark-text-secondary')}>
                    {c.name}
                    {isDom && <span className="ml-1 text-[9px] text-term-amber">◀ dominant</span>}
                  </span>
                  <div className="flex-1 h-2 bg-dark-surface-elevated/40 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${pct}%`, backgroundColor: c.color, opacity: 0.85 }}
                    />
                  </div>
                  <span className="w-16 text-right tabular-nums text-dark-text-primary">
                    {formatBig(c.count)}
                  </span>
                  <span className="w-12 text-right tabular-nums text-dark-text-disabled">
                    {pct.toFixed(1)}%
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Right: scale intuition */}
        <div className="flex flex-col gap-2 min-w-0 overflow-auto">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            scale intuition
          </div>
          <div className="rounded border border-dark-border bg-dark-surface-elevated/40 p-3 font-mono text-[10.5px] leading-relaxed text-dark-text-muted">
            <div>
              attention Q/K/V: <span className="text-term-cyan">3 · L · d²</span>
            </div>
            <div>
              MLP up + down: <span className="text-term-pink">8 · L · d²</span>
            </div>
            <div>
              embedding: <span className="text-term-teal">V · d</span>
            </div>
            <div className="mt-2 text-dark-text-disabled">
              per-block d² terms = 12 · d². embedding is linear in d. the crossover:{' '}
              <span className="text-term-amber">L · 12d² &gt; V · d</span> at{' '}
              <span className="text-term-amber tabular-nums">d &gt; V / (12L)</span>.
            </div>
            <div className="mt-2 text-dark-text-disabled">
              at d={d} · L={L} · V={formatBig(V)}: crossover d = {Math.round(V / (12 * L))}. you are{' '}
              <span className={d > V / (12 * L) ? 'text-term-pink' : 'text-term-teal'}>
                {d > V / (12 * L) ? 'MLP-dominated' : 'embedding-dominated'}
              </span>
              .
            </div>
          </div>

          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-2">
            known checkpoints
          </div>
          <div className="flex flex-col gap-1 font-mono text-[10.5px]">
            <KnownRow name="GPT-2 small" d={768} L={12} V={50257} current={[d, L, V]} onClick={() => { setD(768); setL(12); setV(50257) }} />
            <KnownRow name="GPT-2 medium" d={1024} L={24} V={50257} current={[d, L, V]} onClick={() => { setD(1024); setL(24); setV(50257) }} />
            <KnownRow name="GPT-2 XL-ish" d={2048} L={48} V={50257} current={[d, L, V]} onClick={() => { setD(2048); setL(48); setV(50257) }} />
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function PresetRow({
  label,
  options,
  value,
  onChange,
  fmt,
}: {
  label: string
  options: number[]
  value: number
  onChange: (v: number) => void
  fmt: (v: number) => string
}) {
  return (
    <div className="flex items-center gap-2 font-mono text-[11px]">
      <span className="text-dark-text-secondary">{label}</span>
      <div className="flex items-center gap-0.5">
        {options.map((o) => (
          <button
            key={o}
            onClick={() => onChange(o)}
            className={cn(
              'px-2 py-0.5 rounded text-[10.5px] transition-all',
              value === o
                ? 'bg-dark-accent text-white'
                : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
            )}
          >
            {fmt(o)}
          </button>
        ))}
      </div>
    </div>
  )
}

function KnownRow({
  name,
  d,
  L,
  V,
  current,
  onClick,
}: {
  name: string
  d: number
  L: number
  V: number
  current: [number, number, number]
  onClick: () => void
}) {
  const isActive = current[0] === d && current[1] === L && current[2] === V
  return (
    <button
      onClick={onClick}
      className={cn(
        'text-left px-2 py-1 rounded transition-all',
        isActive
          ? 'bg-term-amber/10 text-term-amber border border-term-amber/40'
          : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
      )}
    >
      <span className="font-mono">{name}</span>{' '}
      <span className="text-dark-text-disabled">
        d={d}, L={L}
      </span>
    </button>
  )
}
