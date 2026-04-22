'use client'

import { useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Stack N transformer blocks vertically. Each block is a slim rectangle that
// renders its per-block parameter count. Sliders control N, d_model, and
// seq_len. A readout panel shows total params, memory at fp16, and FLOPs per
// token. Preset buttons jump to nano-GPT / GPT-2 small / GPT-3 scale, and we
// annotate where famous "scaling-laws" thresholds sit in the stack.

type Preset = {
  name: string
  N: number
  d: number
  T: number
  note: string
}

const PRESETS: Preset[] = [
  { name: 'nano-GPT',   N: 6,  d: 384, T: 256,  note: 'toy — trains in minutes on a laptop GPU.' },
  { name: 'GPT-2 small', N: 12, d: 768, T: 1024, note: '124M params · the original "GPT-2 small".' },
  { name: 'GPT-2 med.',  N: 24, d: 1024, T: 1024, note: '355M params · where depth-effects show up.' },
  { name: 'GPT-3 13B',   N: 40, d: 5120, T: 2048, note: '~13B params · meaningful emergent behaviors.' },
  { name: 'GPT-3 175B',  N: 96, d: 12288, T: 2048, note: '175B params · scaling-laws flagship.' },
]

// Per block: MHA params = 4 d² (QKV + O). MLP params = 2 · d · 4d = 8 d².
// LN params = 4 d (two LayerNorms, each 2d). Bias-free approximation.
function blockParams(d: number): number {
  return 4 * d * d + 8 * d * d + 4 * d
}

function blockFlopsPerToken(d: number, T: number): number {
  // MHA: 4 d² + 2 d T  (amortized per token). MLP: 16 d². Ignore LN.
  return 4 * d * d + 2 * d * T + 16 * d * d
}

function formatN(n: number): string {
  if (n >= 1e12) return (n / 1e12).toFixed(2) + ' T'
  if (n >= 1e9) return (n / 1e9).toFixed(2) + ' B'
  if (n >= 1e6) return (n / 1e6).toFixed(2) + ' M'
  if (n >= 1e3) return (n / 1e3).toFixed(2) + ' K'
  return n.toFixed(0)
}

function formatBytes(b: number): string {
  if (b >= 1e12) return (b / 1e12).toFixed(2) + ' TB'
  if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB'
  if (b >= 1e6) return (b / 1e6).toFixed(2) + ' MB'
  if (b >= 1e3) return (b / 1e3).toFixed(2) + ' KB'
  return b.toFixed(0) + ' B'
}

export default function BlockStack() {
  const [N, setN] = useState(12)
  const [d, setD] = useState(768)
  const [T, setT] = useState(1024)
  const [preset, setPreset] = useState<string | null>('GPT-2 small')

  const perBlock = blockParams(d)
  const embedParams = 50257 * d // GPT-2 vocab approx
  const totalParams = N * perBlock + embedParams
  const totalBytesFp16 = totalParams * 2
  const flopsTok = N * blockFlopsPerToken(d, T)

  const activationBytesFp16 = N * T * d * 2 // approx per-layer residual cache

  const apply = (p: Preset) => {
    setN(p.N); setD(p.d); setT(p.T); setPreset(p.name)
  }

  const activePreset = PRESETS.find((p) => p.name === preset)

  // Scaling-law milestone markers
  const milestones = [
    { at: 6, label: 'emergent in-context learning appears', color: '#67e8f9' },
    { at: 12, label: 'small-GPT range', color: '#a78bfa' },
    { at: 24, label: 'tasks start composing', color: '#fbbf24' },
    { at: 48, label: 'GPT-3 13B regime', color: '#f472b6' },
    { at: 96, label: 'GPT-3 175B', color: '#f87171' },
  ].filter((m) => m.at <= N + 2)

  return (
    <WidgetFrame
      widgetName="BlockStack"
      label="scale the block — stack them deep"
      right={<span className="font-mono">{N} × block · d = {d} · T = {T}</span>}
      aspect="tall"
      controls={
        <div className="flex flex-wrap items-center gap-3">
          {PRESETS.map((p) => (
            <button
              key={p.name}
              onClick={() => apply(p)}
              className={cn(
                'px-2 py-1 rounded text-[10.5px] font-mono transition-all',
                preset === p.name
                  ? 'bg-dark-accent text-white'
                  : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
              )}
            >
              {p.name}
            </button>
          ))}
          <Slider
            label="N"
            value={N}
            min={1}
            max={96}
            step={1}
            onChange={(v) => { setN(Math.round(v)); setPreset(null) }}
            format={(v) => v.toFixed(0)}
            accent="accent-term-purple"
          />
          <div className="ml-auto">
            <Readout label="total" value={formatN(totalParams)} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-1 md:grid-cols-[1fr_220px] gap-4">
        {/* Stack visualization */}
        <div className="relative flex flex-col gap-[2px] overflow-auto pr-2">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-1">
            stack (bottom = input · top = output head)
          </div>
          <div className="flex flex-col-reverse gap-[2px] flex-1 min-h-0">
            {Array.from({ length: N }).map((_, i) => {
              const milestone = milestones.find((m) => m.at === i + 1)
              return (
                <div
                  key={i}
                  className="relative rounded border border-dark-border flex items-center bg-dark-surface-elevated/30"
                  style={{
                    minHeight: `${Math.max(10, Math.min(22, 380 / N))}px`,
                    background: `linear-gradient(90deg, rgba(167,139,250,${0.08 + (i / N) * 0.18}) 0%, rgba(103,232,249,${0.05 + (i / N) * 0.15}) 100%)`,
                  }}
                  title={`block ${i + 1} · ${formatN(perBlock)} params`}
                >
                  <div className="w-10 pl-2 font-mono text-[9px] text-dark-text-disabled">{i + 1}</div>
                  <div className="flex-1 h-1 bg-dark-bg/50 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-term-purple/60"
                      style={{ width: `${Math.min(100, (perBlock / 1e8) * 100)}%` }}
                    />
                  </div>
                  <div className="w-24 px-2 text-right font-mono text-[9.5px] text-dark-text-secondary tabular-nums">
                    {formatN(perBlock)}
                  </div>
                  {milestone && (
                    <div
                      className="absolute -right-1 top-1/2 -translate-y-1/2 translate-x-full whitespace-nowrap font-mono text-[9px] px-1.5 py-0.5 rounded border ml-2"
                      style={{ color: milestone.color, borderColor: milestone.color, backgroundColor: '#0f0f1a' }}
                    >
                      ← {milestone.label}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
          <div className="text-[9.5px] font-mono text-dark-text-disabled mt-1 italic">
            per-block params = 12·d² + 4d = {formatN(perBlock)}
          </div>
        </div>

        {/* Readouts */}
        <div className="flex flex-col gap-3 min-w-0">
          <div className="rounded border border-dark-border p-3 bg-dark-surface-elevated/30">
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
              total parameters
            </div>
            <div className="text-[22px] font-mono text-term-amber tabular-nums">
              {formatN(totalParams)}
            </div>
            <div className="text-[10px] font-mono text-dark-text-muted mt-1">
              blocks: {formatN(N * perBlock)} · embedding: {formatN(embedParams)}
            </div>
          </div>

          <div className="rounded border border-dark-border p-3 bg-dark-surface-elevated/30">
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
              memory footprint
            </div>
            <div className="font-mono text-[11.5px] text-dark-text-primary">
              weights (fp16): <span className="text-term-cyan">{formatBytes(totalBytesFp16)}</span>
            </div>
            <div className="font-mono text-[11.5px] text-dark-text-primary">
              activations ≈: <span className="text-term-pink">{formatBytes(activationBytesFp16)}</span>
            </div>
            <div className="font-mono text-[10px] text-dark-text-disabled mt-1 italic">
              (activation estimate for one forward pass.)
            </div>
          </div>

          <div className="rounded border border-dark-border p-3 bg-dark-surface-elevated/30">
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
              compute per token
            </div>
            <div className="text-[16px] font-mono text-term-green tabular-nums">
              {formatN(flopsTok)} FLOPs
            </div>
            <div className="text-[10px] font-mono text-dark-text-muted mt-1">
              per-token ≈ 2·N·params (a useful rule of thumb).
            </div>
          </div>

          {activePreset && (
            <div className="rounded border border-dark-border p-3" style={{ backgroundColor: 'rgba(167,139,250,0.08)' }}>
              <div className="text-[10px] font-mono uppercase tracking-wider text-term-purple mb-1">
                preset: {activePreset.name}
              </div>
              <p className="font-mono text-[11px] text-dark-text-primary leading-relaxed">
                {activePreset.note}
              </p>
            </div>
          )}
        </div>
      </div>
    </WidgetFrame>
  )
}
