'use client'

import { useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Assemble an LSTM update by hand. Sliders for f (forget), i (input),
// g (candidate in [-1,1]), o (output), and c_{t-1} (prior cell state). The
// widget computes c_t = f·c_{t-1} + i·g and h_t = o·tanh(c_t), and draws a
// horizontal-bar breakdown so it's visually clear which term did what. A few
// "preset" buttons jump to instructive gate configurations.

type Preset = {
  name: string
  f: number
  i: number
  g: number
  o: number
  c_prev: number
  description: string
}

const PRESETS: Preset[] = [
  {
    name: 'preserve',
    f: 1,
    i: 0,
    g: 0,
    o: 1,
    c_prev: 0.7,
    description: 'f=1, i=0 — the conveyor belt just runs. c_t = c_(t-1).',
  },
  {
    name: 'overwrite',
    f: 0,
    i: 1,
    g: -0.8,
    o: 1,
    c_prev: 0.7,
    description: 'f=0, i=1 — wipe the past, write the candidate fresh.',
  },
  {
    name: 'silent',
    f: 1,
    i: 0.2,
    g: 0.5,
    o: 0,
    c_prev: 0.3,
    description: 'o=0 — the cell remembers, but tells no one (h_t = 0).',
  },
  {
    name: 'mix',
    f: 0.6,
    i: 0.5,
    g: 0.4,
    o: 0.8,
    c_prev: 0.5,
    description: 'normal operation — partial forget, partial write, partial reveal.',
  },
]

export default function LSTMGateViz() {
  const [f, setF] = useState(0.6)
  const [i, setI] = useState(0.5)
  const [g, setG] = useState(0.4)
  const [o, setO] = useState(0.8)
  const [cPrev, setCPrev] = useState(0.5)
  const [preset, setPreset] = useState<string | null>('mix')

  const forgetTerm = f * cPrev
  const inputTerm = i * g
  const c_t = forgetTerm + inputTerm
  const tanhC = Math.tanh(c_t)
  const h_t = o * tanhC

  const applyPreset = (p: Preset) => {
    setF(p.f); setI(p.i); setG(p.g); setO(p.o); setCPrev(p.c_prev); setPreset(p.name)
  }

  const activePreset = PRESETS.find((p) => p.name === preset)

  return (
    <WidgetFrame
      widgetName="LSTMGateViz"
      label="LSTM gate assembly — move sliders to build c_t and h_t"
      right={<span className="font-mono">c_t = f·c_(t-1) + i·g · h_t = o·tanh(c_t)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5 flex-wrap">
            {PRESETS.map((p) => (
              <button
                key={p.name}
                onClick={() => applyPreset(p)}
                className={cn(
                  'px-2 py-1 rounded text-[10.5px] font-mono uppercase transition-all',
                  preset === p.name
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {p.name}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="c_t" value={c_t.toFixed(3)} accent="text-term-amber" />
            <Readout label="h_t" value={h_t.toFixed(3)} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 grid grid-cols-1 md:grid-cols-[280px_1fr] gap-6 overflow-auto">
        {/* Left: sliders for all inputs */}
        <div className="flex flex-col gap-3 min-w-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            inputs
          </div>
          <GateSlider label="f_t  forget" color="#f87171" value={f} min={0} max={1} onChange={(v) => { setF(v); setPreset(null) }} />
          <GateSlider label="i_t  input" color="#67e8f9" value={i} min={0} max={1} onChange={(v) => { setI(v); setPreset(null) }} />
          <GateSlider label="g_t  candidate" color="#a78bfa" value={g} min={-1} max={1} onChange={(v) => { setG(v); setPreset(null) }} bipolar />
          <GateSlider label="o_t  output" color="#4ade80" value={o} min={0} max={1} onChange={(v) => { setO(v); setPreset(null) }} />
          <GateSlider label="c_(t-1)" color="#fbbf24" value={cPrev} min={-2} max={2} onChange={(v) => { setCPrev(v); setPreset(null) }} bipolar />
          {activePreset && (
            <div className="mt-2 text-[11px] font-mono text-dark-text-muted leading-snug">
              <span className="text-term-amber">preset:</span> {activePreset.description}
            </div>
          )}
        </div>

        {/* Right: breakdown */}
        <div className="flex flex-col gap-4 min-w-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            c_t = f · c_(t-1)  +  i · g
          </div>

          <TermBar
            label="f · c_(t-1)"
            value={forgetTerm}
            compose={`${f.toFixed(2)} × ${cPrev.toFixed(2)}`}
            color="#f87171"
            max={2}
          />
          <TermBar
            label="i · g"
            value={inputTerm}
            compose={`${i.toFixed(2)} × ${g.toFixed(2)}`}
            color="#a78bfa"
            max={2}
          />
          <div className="border-t border-dark-border my-1" />
          <TermBar
            label="c_t (sum)"
            value={c_t}
            compose={`${forgetTerm.toFixed(2)} + ${inputTerm.toFixed(2)}`}
            color="#fbbf24"
            max={2}
            highlight
          />

          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-3">
            h_t = o · tanh(c_t)
          </div>
          <TermBar
            label="tanh(c_t)"
            value={tanhC}
            compose={`tanh(${c_t.toFixed(2)})`}
            color="#60a5fa"
            max={1}
          />
          <TermBar
            label="h_t"
            value={h_t}
            compose={`${o.toFixed(2)} × ${tanhC.toFixed(2)}`}
            color="#4ade80"
            max={1}
            highlight
          />
        </div>
      </div>
    </WidgetFrame>
  )
}

function GateSlider({
  label,
  color,
  value,
  min,
  max,
  onChange,
  bipolar,
}: {
  label: string
  color: string
  value: number
  min: number
  max: number
  onChange: (v: number) => void
  bipolar?: boolean
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between font-mono text-[11px]">
        <span className="text-dark-text-secondary">{label}</span>
        <span className="tabular-nums" style={{ color }}>
          {value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={0.01}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1 rounded-full bg-dark-border cursor-pointer"
        style={{ accentColor: color }}
      />
      <div className="relative h-2 bg-dark-surface-elevated/40 rounded-sm overflow-hidden">
        <div
          className="absolute top-0 bottom-0"
          style={{
            left: bipolar ? '50%' : 0,
            width: bipolar
              ? `${(Math.abs(value) / Math.max(Math.abs(min), Math.abs(max))) * 50}%`
              : `${((value - min) / (max - min)) * 100}%`,
            backgroundColor: color,
            opacity: 0.65,
            transform: bipolar && value < 0 ? 'translateX(-100%)' : undefined,
          }}
        />
      </div>
    </div>
  )
}

function TermBar({
  label,
  value,
  compose,
  color,
  max,
  highlight,
}: {
  label: string
  value: number
  compose: string
  color: string
  max: number
  highlight?: boolean
}) {
  const pct = Math.min(1, Math.abs(value) / max)
  return (
    <div className={cn('flex flex-col gap-1', highlight && 'bg-dark-surface-elevated/40 p-2 -mx-2 rounded')}>
      <div className="flex items-center justify-between font-mono text-[11px]">
        <span className="text-dark-text-secondary">{label}</span>
        <span className="flex items-center gap-2">
          <span className="text-dark-text-disabled">{compose}</span>
          <span className="tabular-nums w-14 text-right" style={{ color }}>
            {value.toFixed(3)}
          </span>
        </span>
      </div>
      <div className="relative h-3 bg-dark-surface-elevated/40 rounded-sm overflow-hidden">
        <div className="absolute top-0 bottom-0 border-l border-dark-border/70" style={{ left: '50%' }} />
        <div
          className="absolute top-0 bottom-0"
          style={{
            left: value >= 0 ? '50%' : `${50 - pct * 50}%`,
            width: `${pct * 50}%`,
            backgroundColor: color,
            opacity: 0.75,
          }}
        />
      </div>
    </div>
  )
}
