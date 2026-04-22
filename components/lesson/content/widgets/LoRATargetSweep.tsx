'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Pick which linear layers to target with LoRA adapters. Each module has a
// fixed d-in, d-out (we use 4096 for Q/K/V/O and 4096 → 11008 for the
// MLP gate/up/down in Llama-2 7B style). Trainable params computed as
// 2 · r · (d_in + d_out) · num_layers. A small curated "accuracy" table
// matches the sweep in the LoRA paper appendix and Hu et al. ablations —
// attention-only is the baseline, Q+V the common sweet spot, all-linear
// gives a small bump but at 3× the cost.

const R = 16
const LAYERS = 32

type ModuleKey = 'q' | 'k' | 'v' | 'o' | 'gate' | 'up' | 'down'

const MODULES: { key: ModuleKey; label: string; dIn: number; dOut: number; group: 'attn' | 'mlp' }[] = [
  { key: 'q', label: 'q_proj', dIn: 4096, dOut: 4096, group: 'attn' },
  { key: 'k', label: 'k_proj', dIn: 4096, dOut: 4096, group: 'attn' },
  { key: 'v', label: 'v_proj', dIn: 4096, dOut: 4096, group: 'attn' },
  { key: 'o', label: 'o_proj', dIn: 4096, dOut: 4096, group: 'attn' },
  { key: 'gate', label: 'gate_proj', dIn: 4096, dOut: 11008, group: 'mlp' },
  { key: 'up', label: 'up_proj', dIn: 4096, dOut: 11008, group: 'mlp' },
  { key: 'down', label: 'down_proj', dIn: 11008, dOut: 4096, group: 'mlp' },
]

type Selection = Record<ModuleKey, boolean>

const PRESETS: { name: string; sel: Partial<Selection>; accuracy: number; note: string }[] = [
  { name: 'Q only', sel: { q: true }, accuracy: 71.1, note: 'under-capacity — too few degrees of freedom' },
  { name: 'Q + V (common)', sel: { q: true, v: true }, accuracy: 73.9, note: 'the LoRA paper sweet spot' },
  { name: 'attention only', sel: { q: true, k: true, v: true, o: true }, accuracy: 72.3, note: 'classic baseline' },
  { name: 'MLP only', sel: { gate: true, up: true, down: true }, accuracy: 73.2, note: 'larger d_out helps' },
  { name: 'all-linear', sel: { q: true, k: true, v: true, o: true, gate: true, up: true, down: true }, accuracy: 74.1, note: 'every linear layer — best raw perf, 3× cost' },
]

function fakeAccuracy(sel: Selection): { acc: number; matched: string | null } {
  // Try to match a preset. Otherwise, sum a weighted contribution.
  for (const p of PRESETS) {
    const match = MODULES.every((m) => (sel[m.key] === !!p.sel[m.key]))
    if (match) return { acc: p.accuracy, matched: p.name }
  }
  // Per-module contributions (empirically, v_proj and down_proj carry more).
  const weights: Record<ModuleKey, number> = { q: 0.8, k: 0.4, v: 1.1, o: 0.7, gate: 0.8, up: 0.8, down: 1.0 }
  const base = 69.4
  let bonus = 0
  MODULES.forEach((m) => {
    if (sel[m.key]) bonus += weights[m.key]
  })
  return { acc: Math.min(74.5, base + bonus * 0.6), matched: null }
}

export default function LoRATargetSweep() {
  const [sel, setSel] = useState<Selection>({ q: true, v: true, k: false, o: false, gate: false, up: false, down: false })

  const trainable = useMemo(() => {
    let total = 0
    MODULES.forEach((m) => {
      if (sel[m.key]) total += 2 * R * (m.dIn + m.dOut) * LAYERS
    })
    return total
  }, [sel])

  const { acc, matched } = useMemo(() => fakeAccuracy(sel), [sel])
  const toggle = (k: ModuleKey) => setSel((s) => ({ ...s, [k]: !s[k] }))
  const applyPreset = (p: (typeof PRESETS)[number]) => {
    const next: Selection = { q: false, k: false, v: false, o: false, gate: false, up: false, down: false }
    Object.entries(p.sel).forEach(([k, v]) => {
      if (v) next[k as ModuleKey] = true
    })
    setSel(next)
  }

  const isQV = matched === 'Q + V (common)'

  return (
    <WidgetFrame
      widgetName="LoRATargetSweep"
      label="LoRA target modules — which linears get adapters?"
      right={<span className="font-mono">r = {R} · {LAYERS} layers · Llama-2 7B shapes</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5 flex-wrap">
            {PRESETS.map((p) => (
              <button
                key={p.name}
                onClick={() => applyPreset(p)}
                className={cn(
                  'px-2 py-1 rounded text-[10.5px] font-mono transition-all border',
                  matched === p.name
                    ? p.name === 'Q + V (common)'
                      ? 'border-term-amber text-term-amber bg-term-amber/10'
                      : 'border-dark-accent text-dark-accent bg-dark-accent/10'
                    : 'border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {p.name}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="trainable" value={trainable.toLocaleString('en-US')} accent="text-term-green" />
            <Readout label="accuracy" value={`${acc.toFixed(1)}%`} accent={isQV ? 'text-term-amber' : 'text-term-cyan'} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[1fr_260px] gap-4 overflow-hidden">
        {/* Left: module toggles */}
        <div className="flex flex-col gap-3 min-w-0 overflow-auto">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            attention block
          </div>
          <div className="grid grid-cols-4 gap-2">
            {MODULES.filter((m) => m.group === 'attn').map((m) => (
              <ModuleToggle key={m.key} m={m} on={sel[m.key]} onClick={() => toggle(m.key)} />
            ))}
          </div>
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-2">
            MLP block
          </div>
          <div className="grid grid-cols-3 gap-2">
            {MODULES.filter((m) => m.group === 'mlp').map((m) => (
              <ModuleToggle key={m.key} m={m} on={sel[m.key]} onClick={() => toggle(m.key)} />
            ))}
          </div>

          <div className="mt-3 font-mono text-[10.5px] text-dark-text-muted leading-snug border-t border-dark-border pt-3">
            {matched ? (
              <span>
                <span className="text-term-amber">preset:</span> {PRESETS.find((p) => p.name === matched)?.note}
              </span>
            ) : (
              <span>
                <span className="text-term-cyan">custom mix:</span> toggle presets on the top bar to land on standard configurations.
              </span>
            )}
          </div>
        </div>

        {/* Right: readout */}
        <div className="flex flex-col gap-3 min-w-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            budget
          </div>
          <div className="font-mono text-[11px] bg-dark-surface-elevated/40 rounded p-3 space-y-1.5">
            <div className="flex justify-between">
              <span className="text-dark-text-secondary">per-adapter</span>
              <span className="text-dark-text-primary tabular-nums">2 · r · (d_in + d_out)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-dark-text-secondary">× layers</span>
              <span className="text-dark-text-primary tabular-nums">× {LAYERS}</span>
            </div>
            <div className="border-t border-dark-border my-1.5" />
            <div className="flex justify-between">
              <span className="text-term-green">trainable</span>
              <span className="text-term-green tabular-nums">{trainable.toLocaleString('en-US')}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-dark-text-disabled">vs full 7B</span>
              <span className="text-dark-text-disabled tabular-nums">
                {((trainable / 6.74e9) * 100).toFixed(3)}%
              </span>
            </div>
          </div>

          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-2">
            task accuracy (MMLU-ish)
          </div>
          <div className="relative h-6 bg-dark-surface-elevated/40 rounded overflow-hidden">
            <div
              className={cn('absolute inset-y-0 left-0', isQV ? 'bg-term-amber/70' : 'bg-term-cyan/70')}
              style={{ width: `${((acc - 68) / (74.5 - 68)) * 100}%` }}
            />
            <div className="absolute inset-0 flex items-center justify-center font-mono text-[11px] text-white">
              {acc.toFixed(1)}%
            </div>
          </div>
          <div className="font-mono text-[10px] text-dark-text-disabled">
            full fine-tune reference: 74.8%
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function ModuleToggle({
  m,
  on,
  onClick,
}: {
  m: { key: ModuleKey; label: string; dIn: number; dOut: number }
  on: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'flex flex-col items-start gap-0.5 px-2 py-2 rounded border font-mono text-[10.5px] transition-all text-left',
        on
          ? 'border-term-cyan bg-term-cyan/10 text-term-cyan'
          : 'border-dark-border bg-dark-surface-elevated/40 text-dark-text-secondary hover:text-dark-text-primary hover:border-dark-border-hover',
      )}
    >
      <span className="font-semibold">{m.label}</span>
      <span className="text-[9.5px] text-dark-text-disabled">
        {m.dIn} × {m.dOut}
      </span>
    </button>
  )
}
