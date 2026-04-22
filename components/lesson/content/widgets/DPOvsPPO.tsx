'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Side-by-side comparison of PPO vs DPO. PPO needs 4 models loaded at once:
// policy, value head, reward model, reference. DPO needs only policy + ref.
// We visualize each: model cards stacked with their memory cost, a bar for
// per-step wall-clock time, and a metric-compare row ( GRUvsLSTM-style )
// summarizing GPU memory and step time. Slider for model size in billions.

type ModelBlock = {
  role: string
  color: string
  scale: number // fraction of base for this role
}

const PPO_MODELS: ModelBlock[] = [
  { role: 'policy π_θ', color: 'amber', scale: 1.0 },
  { role: 'reference π_ref', color: 'cyan', scale: 1.0 },
  { role: 'reward model', color: 'purple', scale: 1.0 },
  { role: 'value V_φ', color: 'green', scale: 1.0 },
]

const DPO_MODELS: ModelBlock[] = [
  { role: 'policy π_θ', color: 'amber', scale: 1.0 },
  { role: 'reference π_ref', color: 'cyan', scale: 1.0 },
]

const TERM = {
  amber: 'bg-term-amber',
  cyan: 'bg-term-cyan',
  purple: 'bg-term-purple',
  green: 'bg-term-green',
} as const

export default function DPOvsPPO() {
  const [sizeB, setSizeB] = useState(7)

  // Approx fp16 GB per model = 2 × params_B (for weights alone).
  // Full trainable with Adam: roughly 6× for the one being trained.
  // PPO trains policy + value; DPO trains only policy.
  const { ppoMem, dpoMem, ppoStepMs, dpoStepMs } = useMemo(() => {
    const weightsGB = 2 * sizeB // fp16 weights
    const trainableOverhead = 6 * sizeB // Adam + grads in fp32 ≈ 6× params
    const ppo =
      weightsGB * 4 + // four models loaded
      trainableOverhead * 2 // policy + value are trainable
    const dpo =
      weightsGB * 2 + // policy + ref
      trainableOverhead * 1 // policy only
    // Step time: dominated by forwards. PPO does rollout + RM eval + value + 4 forwards
    // per step. DPO does chosen + rejected forward under π_θ and π_ref = 4 forwards total.
    const ppoMs = 420 + sizeB * 50
    const dpoMs = 180 + sizeB * 18
    return { ppoMem: ppo, dpoMem: dpo, ppoStepMs: ppoMs, dpoStepMs: dpoMs }
  }, [sizeB])

  const metrics = [
    {
      label: 'GPU memory',
      ppoVal: ppoMem,
      dpoVal: dpoMem,
      fmt: (v: number) => `${v.toFixed(0)} GB`,
      dpoWins: true,
    },
    {
      label: 'wall-clock per step',
      ppoVal: ppoStepMs,
      dpoVal: dpoStepMs,
      fmt: (v: number) => `${v.toFixed(0)} ms`,
      dpoWins: true,
    },
    {
      label: 'models loaded',
      ppoVal: 4,
      dpoVal: 2,
      fmt: (v: number) => `${v.toFixed(0)} models`,
      dpoWins: true,
    },
  ]

  return (
    <WidgetFrame
      widgetName="DPOvsPPO"
      label="DPO vs. PPO — what do you actually pay for?"
      right={<span className="font-mono">loaded models · GB · ms/step</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="model size (B)"
            value={sizeB}
            min={1}
            max={70}
            step={1}
            onChange={(v) => setSizeB(Math.round(v))}
            format={(v) => `${Math.round(v)}B`}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="PPO mem" value={`${ppoMem.toFixed(0)} GB`} accent="text-term-rose" />
            <Readout label="DPO mem" value={`${dpoMem.toFixed(0)} GB`} accent="text-term-green" />
            <Readout label="speedup" value={`${(ppoStepMs / dpoStepMs).toFixed(2)}×`} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-[1fr_1fr_auto] gap-4 overflow-hidden">
        {/* PPO column */}
        <ModelColumn
          title="PPO pipeline"
          totalGB={ppoMem}
          stepMs={ppoStepMs}
          models={PPO_MODELS}
          sizeB={sizeB}
          highlight={false}
          caption="rollout → reward → advantage → clipped surrogate. Four models live on GPU; two are trained."
        />
        <ModelColumn
          title="DPO pipeline"
          totalGB={dpoMem}
          stepMs={dpoStepMs}
          models={DPO_MODELS}
          sizeB={sizeB}
          highlight={true}
          caption="closed-form from preferences. Two models; only the policy learns. No RM, no value head."
        />

        {/* Right: metric bars */}
        <div className="flex flex-col gap-3 min-w-0 w-[240px]">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            head-to-head
          </div>
          {metrics.map((m) => (
            <MetricRow key={m.label} {...m} />
          ))}

          <div className="mt-2 font-mono text-[10.5px] text-dark-text-muted leading-snug border-t border-dark-border pt-3">
            <span className="text-term-amber">trade-off:</span> PPO&apos;s extra infrastructure pays off when reward
            modeling needs to extrapolate far from the SFT data; DPO wins when you have enough preferences and want
            simplicity.
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function ModelColumn({
  title,
  totalGB,
  stepMs,
  models,
  sizeB,
  highlight,
  caption,
}: {
  title: string
  totalGB: number
  stepMs: number
  models: ModelBlock[]
  sizeB: number
  highlight: boolean
  caption: string
}) {
  return (
    <div className={cn(
      'flex flex-col gap-2 rounded border p-3 min-w-0',
      highlight ? 'border-term-green/40 bg-term-green/5' : 'border-dark-border bg-dark-surface-elevated/20',
    )}>
      <div className="flex items-center justify-between font-mono text-[10.5px]">
        <span className={cn('uppercase tracking-wider', highlight ? 'text-term-green' : 'text-dark-text-secondary')}>
          {title}
        </span>
        <span className="text-dark-text-muted tabular-nums">{stepMs.toFixed(0)} ms/step</span>
      </div>
      <div className="flex flex-col gap-1.5 font-mono text-[11px]">
        {models.map((m) => (
          <div key={m.role} className="flex items-center gap-2">
            <span className={cn('inline-block w-3 h-3 rounded-sm', TERM[m.color as keyof typeof TERM])} />
            <span className="text-dark-text-primary flex-1">{m.role}</span>
            <span className="tabular-nums text-dark-text-disabled">{sizeB}B</span>
          </div>
        ))}
      </div>

      <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-2">
        stacked memory
      </div>
      <div className="relative h-6 bg-dark-surface-elevated/40 rounded overflow-hidden flex">
        {models.map((m, i) => {
          const weightW = (2 * sizeB * m.scale / totalGB) * 100
          return (
            <div
              key={i}
              className={cn(TERM[m.color as keyof typeof TERM], 'opacity-80')}
              style={{ width: `${weightW}%` }}
              title={`${m.role}: ${(2 * sizeB).toFixed(0)} GB`}
            />
          )
        })}
      </div>
      <div className="flex items-center justify-between font-mono text-[10px] text-dark-text-muted">
        <span>{models.length} models</span>
        <span className="tabular-nums">≈{totalGB.toFixed(0)} GB</span>
      </div>

      <div className="mt-2 text-[10.5px] font-mono text-dark-text-muted leading-snug">
        {caption}
      </div>
    </div>
  )
}

function MetricRow({
  label,
  ppoVal,
  dpoVal,
  fmt,
  dpoWins,
}: {
  label: string
  ppoVal: number
  dpoVal: number
  fmt: (v: number) => string
  dpoWins: boolean
}) {
  const max = Math.max(ppoVal, dpoVal, 1)
  return (
    <div className="flex flex-col gap-1">
      <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-secondary">
        {label}
      </div>
      <div className="flex items-center gap-2 font-mono text-[10.5px]">
        <span className="w-10 text-term-rose">PPO</span>
        <div className="flex-1 h-4 bg-dark-surface-elevated/40 rounded overflow-hidden">
          <div className="h-full bg-term-rose/70" style={{ width: `${(ppoVal / max) * 100}%` }} />
        </div>
        <span className="w-16 text-right tabular-nums text-dark-text-primary">{fmt(ppoVal)}</span>
      </div>
      <div className="flex items-center gap-2 font-mono text-[10.5px]">
        <span className="w-10 text-term-green">DPO</span>
        <div className="flex-1 h-4 bg-dark-surface-elevated/40 rounded overflow-hidden">
          <div className={cn('h-full', dpoWins ? 'bg-term-green' : 'bg-term-green/70')} style={{ width: `${(dpoVal / max) * 100}%` }} />
        </div>
        <span className={cn('w-16 text-right tabular-nums', dpoWins ? 'text-term-green' : 'text-dark-text-primary')}>{fmt(dpoVal)}</span>
      </div>
    </div>
  )
}
