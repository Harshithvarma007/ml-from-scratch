'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Button, Readout } from './WidgetFrame'
import { RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'

// A grid of neurons, each with random weights and a bias. For a batch of
// inputs, a neuron is "active" if ReLU(w·x + b) > 0 for at least one example.
// If it's never active, it's dead — the gradient is zero and training cannot
// revive it. Leaky ReLU flips the switch: even when the pre-activation is
// negative, the derivative is 0.1, so nothing ever truly dies.

const GRID = 12 // 12×12 = 144 neurons
const BATCH = 64
const INPUT_DIM = 4

type ActName = 'relu' | 'leaky-relu'

interface Neuron {
  w: number[]
  b: number
  activations: number // count of batch items with positive pre-activation
}

// Deterministic pseudo-random (mulberry32) so a given seed + bias gives the
// same grid — keeps the widget reproducible across renders.
function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gauss(rng: () => number): number {
  let u = 0
  let v = 0
  while (u === 0) u = rng()
  while (v === 0) v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function buildGrid(seed: number, biasMean: number): { neurons: Neuron[]; batch: number[][] } {
  const rng = mulberry32(seed)
  const batch: number[][] = []
  for (let i = 0; i < BATCH; i++) {
    const row: number[] = []
    for (let j = 0; j < INPUT_DIM; j++) row.push(gauss(rng))
    batch.push(row)
  }
  const neurons: Neuron[] = []
  for (let i = 0; i < GRID * GRID; i++) {
    const w: number[] = []
    for (let j = 0; j < INPUT_DIM; j++) w.push(gauss(rng) * 0.5)
    const b = biasMean + 0.2 * gauss(rng)
    let activations = 0
    for (const x of batch) {
      let z = b
      for (let j = 0; j < INPUT_DIM; j++) z += w[j] * x[j]
      if (z > 0) activations += 1
    }
    neurons.push({ w, b, activations })
  }
  return { neurons, batch }
}

export default function DeadReluField() {
  const [bias, setBias] = useState(-1.2)
  const [seed, setSeed] = useState(1)
  const [act, setAct] = useState<ActName>('relu')

  const { neurons } = useMemo(() => buildGrid(seed, bias), [seed, bias])

  // For ReLU: dead if never positive. For LReLU: still technically "quiet" but
  // gradients flow, so the cells never go fully black.
  const deadCount = neurons.filter((n) => n.activations === 0).length
  const aliveCount = GRID * GRID - deadCount

  return (
    <WidgetFrame
      widgetName="DeadReluField"
      label="dead neurons — the ReLU failure mode"
      right={
        <>
          <span className="font-mono">12×12 neurons</span>
          <span className="text-dark-text-disabled">·</span>
          <span className="font-mono">batch of {BATCH}</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            {(['relu', 'leaky-relu'] as ActName[]).map((a) => (
              <button
                key={a}
                onClick={() => setAct(a)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono uppercase tracking-wider transition-all',
                  act === a
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
                )}
              >
                {a === 'relu' ? 'ReLU' : 'Leaky ReLU'}
              </button>
            ))}
          </div>
          <Slider
            label="bias μ"
            value={bias}
            min={-3}
            max={0.5}
            step={0.05}
            onChange={setBias}
            accent="accent-term-amber"
          />
          <Button onClick={() => setSeed((s) => s + 1)}>
            <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reseed
          </Button>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="alive" value={String(aliveCount)} accent="text-term-green" />
            <Readout
              label="dead"
              value={String(deadCount)}
              accent={deadCount > 0 ? 'text-term-rose' : 'text-dark-text-disabled'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 flex items-center justify-center p-6">
        <div
          className="grid gap-[2px] w-full h-full max-w-[540px] max-h-[380px]"
          style={{ gridTemplateColumns: `repeat(${GRID}, 1fr)` }}
        >
          {neurons.map((n, i) => {
            const rate = n.activations / BATCH
            // ReLU: rate==0 → pitch black (dead). Positive rate → amber glow.
            // Leaky ReLU: never truly dead, so even rate==0 shows a dim rose
            // tint (gradient is 0.1, neuron is "asleep" but recoverable).
            let bg: string
            let border: string
            if (act === 'relu') {
              if (rate === 0) {
                bg = 'rgba(10, 10, 10, 1)'
                border = 'rgba(244, 63, 94, 0.5)'
              } else {
                const a = 0.2 + rate * 0.8
                bg = `rgba(251, 191, 36, ${a})`
                border = `rgba(251, 191, 36, ${Math.min(1, a + 0.2)})`
              }
            } else {
              if (rate === 0) {
                // Sleepy but alive — dim rose
                bg = 'rgba(244, 63, 94, 0.18)'
                border = 'rgba(244, 63, 94, 0.4)'
              } else {
                const a = 0.2 + rate * 0.8
                bg = `rgba(251, 191, 36, ${a})`
                border = `rgba(251, 191, 36, ${Math.min(1, a + 0.2)})`
              }
            }
            return (
              <div
                key={i}
                className="rounded-sm transition-all"
                style={{
                  backgroundColor: bg,
                  boxShadow: `inset 0 0 0 1px ${border}`,
                }}
                title={`neuron #${i} — active on ${n.activations}/${BATCH}`}
              />
            )
          })}
        </div>
      </div>
      <div className="absolute bottom-2 left-4 right-4 text-[11px] font-mono text-dark-text-muted pointer-events-none flex items-center gap-4">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 rounded-sm bg-term-amber/60" />
          active
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 rounded-sm bg-dark-bg border border-term-rose/50" />
          dead (ReLU — unrecoverable)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 rounded-sm bg-term-rose/20 border border-term-rose/40" />
          asleep (Leaky ReLU — still learning)
        </span>
      </div>
    </WidgetFrame>
  )
}
