'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Stacked memory bars for fine-tuning setups. Each config lists: weights,
// gradients, optimizer state (Adam = 2× weights in fp32), activations (scales
// with batch × seq), LoRA adapter block. We compare against a chosen GPU
// budget (24 / 48 / 80 GB) so you can see at a glance which setups OOM.

type Setup = {
  key: string
  label: string
  weightsGB: number
  gradsGB: number
  optGB: number
  actsGB: number
  adapterGB: number
}

// 7B params ≈ 14GB fp16 weights. Gradients same. Adam = 12GB (2× fp32 moments / 2 = 12 at fp16 or 28 at fp32; use Adam-fp32 ≈ 28).
// QLoRA: weights 4-bit ≈ 3.5GB. LoRA adapters only ~0.1GB. No full gradients.
// 70B: scale up 10×.

const SETUPS: Setup[] = [
  {
    key: '7B full',
    label: '7B full fine-tune',
    weightsGB: 14,
    gradsGB: 14,
    optGB: 28,
    actsGB: 8,
    adapterGB: 0,
  },
  {
    key: '7B LoRA',
    label: '7B LoRA (r=16)',
    weightsGB: 14,
    gradsGB: 0.1,
    optGB: 0.2,
    actsGB: 6,
    adapterGB: 0.1,
  },
  {
    key: '7B QLoRA',
    label: '7B QLoRA',
    weightsGB: 3.5,
    gradsGB: 0.1,
    optGB: 0.2,
    actsGB: 6,
    adapterGB: 0.1,
  },
  {
    key: '70B LoRA',
    label: '70B LoRA (r=16)',
    weightsGB: 140,
    gradsGB: 1.0,
    optGB: 2.0,
    actsGB: 28,
    adapterGB: 1.0,
  },
  {
    key: '70B QLoRA',
    label: '70B QLoRA',
    weightsGB: 35,
    gradsGB: 1.0,
    optGB: 2.0,
    actsGB: 28,
    adapterGB: 1.0,
  },
]

const GPUS: { name: string; gb: number }[] = [
  { name: 'RTX 4090 / A10 (24 GB)', gb: 24 },
  { name: 'L40S / A6000 (48 GB)', gb: 48 },
  { name: 'A100 80G / H100', gb: 80 },
]

const COMPONENTS = [
  { key: 'weightsGB' as const, label: 'weights', color: 'bg-term-amber' },
  { key: 'gradsGB' as const, label: 'gradients', color: 'bg-term-rose' },
  { key: 'optGB' as const, label: 'optimizer state', color: 'bg-term-purple' },
  { key: 'actsGB' as const, label: 'activations', color: 'bg-term-cyan' },
  { key: 'adapterGB' as const, label: 'LoRA adapters', color: 'bg-term-green' },
]

export default function MemoryBudget() {
  const [setupIdx, setSetupIdx] = useState(2)
  const [gpuIdx, setGpuIdx] = useState(0)

  const setup = SETUPS[setupIdx]
  const gpu = GPUS[gpuIdx]
  const totalGB = useMemo(
    () => COMPONENTS.reduce((s, c) => s + setup[c.key], 0),
    [setup],
  )
  const oom = totalGB > gpu.gb
  const usagePct = Math.min(100, (totalGB / gpu.gb) * 100)

  // Render bars for each setup side-by-side for context.
  const maxScale = Math.max(gpu.gb, Math.max(...SETUPS.map((s) => COMPONENTS.reduce((sum, c) => sum + s[c.key], 0))))

  return (
    <WidgetFrame
      widgetName="MemoryBudget"
      label="memory budget — where the GB actually go"
      right={<span className="font-mono">fp16 weights · Adam fp32 moments · activations for seq 2048, bs 1</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5 flex-wrap">
            {SETUPS.map((s, i) => (
              <button
                key={s.key}
                onClick={() => setSetupIdx(i)}
                className={cn(
                  'px-2 py-1 rounded text-[10.5px] font-mono transition-all',
                  setupIdx === i
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {s.key}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="total" value={`${totalGB.toFixed(1)} GB`} accent={oom ? 'text-term-rose' : 'text-term-green'} />
            <Readout label={oom ? 'OOM' : 'usage'} value={`${usagePct.toFixed(0)}%`} accent={oom ? 'text-term-rose' : 'text-term-amber'} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[1fr_260px] gap-4 overflow-hidden">
        {/* Left: bars */}
        <div className="flex flex-col gap-3 min-w-0 overflow-auto">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled flex justify-between">
            <span>setup / components</span>
            <span>scale: 0 → {maxScale.toFixed(0)} GB</span>
          </div>
          <div className="space-y-2">
            {SETUPS.map((s, i) => {
              const total = COMPONENTS.reduce((sum, c) => sum + s[c.key], 0)
              const isActive = i === setupIdx
              const willOOM = total > gpu.gb
              return (
                <div key={s.key} className="flex flex-col gap-1">
                  <div className="flex items-center justify-between font-mono text-[10.5px]">
                    <span className={cn('font-semibold', isActive ? 'text-term-amber' : 'text-dark-text-secondary')}>
                      {s.label}
                    </span>
                    <span className={cn('tabular-nums', willOOM ? 'text-term-rose' : 'text-dark-text-muted')}>
                      {total.toFixed(1)} GB
                    </span>
                  </div>
                  <div className={cn(
                    'relative h-5 bg-dark-surface-elevated/40 rounded overflow-hidden flex',
                    isActive && 'ring-1 ring-term-amber/40',
                  )}>
                    {COMPONENTS.map((c) => {
                      const w = (s[c.key] / maxScale) * 100
                      if (w < 0.2) return null
                      return (
                        <div
                          key={c.key}
                          className={cn(c.color, 'opacity-85')}
                          style={{ width: `${w}%` }}
                          title={`${c.label}: ${s[c.key].toFixed(1)} GB`}
                        />
                      )
                    })}
                    {/* GPU budget line */}
                    <div
                      className="absolute top-0 bottom-0 border-l-2 border-white/40"
                      style={{ left: `${(gpu.gb / maxScale) * 100}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>

          <div className="flex flex-wrap items-center gap-3 pt-2 mt-2 border-t border-dark-border font-mono text-[10px] text-dark-text-muted">
            {COMPONENTS.map((c) => (
              <div key={c.key} className="flex items-center gap-1.5">
                <span className={cn('inline-block w-3 h-3 rounded-sm', c.color)} />
                <span>{c.label}</span>
              </div>
            ))}
            <div className="flex items-center gap-1.5">
              <span className="inline-block w-0.5 h-3 bg-white/50" />
              <span>GPU budget</span>
            </div>
          </div>
        </div>

        {/* Right: GPU selector + callout */}
        <div className="flex flex-col gap-3 min-w-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            target GPU
          </div>
          <div className="flex flex-col gap-1">
            {GPUS.map((g, i) => (
              <button
                key={g.name}
                onClick={() => setGpuIdx(i)}
                className={cn(
                  'text-left px-2.5 py-1.5 rounded font-mono text-[11px] transition-all border',
                  gpuIdx === i
                    ? 'border-term-cyan text-term-cyan bg-term-cyan/10'
                    : 'border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {g.name}
              </button>
            ))}
          </div>

          <div className={cn(
            'mt-2 font-mono text-[10.5px] leading-snug px-3 py-2.5 rounded border',
            oom
              ? 'border-term-rose/40 bg-term-rose/10 text-term-rose'
              : 'border-term-green/40 bg-term-green/5 text-term-green',
          )}>
            {oom ? (
              <>
                <span className="font-semibold">out of memory.</span> {setup.label} needs {totalGB.toFixed(1)} GB, but {gpu.name}
                only has {gpu.gb} GB. Drop the batch, quantize, or rent a bigger chip.
              </>
            ) : (
              <>
                <span className="font-semibold">fits.</span> {setup.label} uses {totalGB.toFixed(1)} / {gpu.gb} GB
                ({usagePct.toFixed(0)}%). Room for a larger batch.
              </>
            )}
          </div>

          <div className="text-[9.5px] font-mono text-dark-text-disabled mt-2 leading-snug">
            Numbers assume fp16 weights, Adam fp32 optimizer (2 moments), activation checkpointing disabled, seq = 2048, batch = 1.
            QLoRA uses NF4 weights (≈4 bits → W/4) and LoRA adapters only train the low-rank matrices.
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
