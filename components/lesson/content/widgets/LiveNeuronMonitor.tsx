'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Button, Readout } from './WidgetFrame'
import { Play, Pause, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'

// A live-training 2 → 32 (ReLU) → 1 MLP on a toy 2D classification task.
// Each hidden neuron gets a cell in a 4×8 grid. The cell's color encodes
// the neuron's activation rate across the current batch — dark when dead,
// bright amber when frequently active. Crank the init scale and watch
// neurons flip off as pre-activations saturate negative across every
// training example.

const HIDDEN = 32
const GRID_COLS = 8
const BATCH = 128

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gauss(rng: () => number): number {
  const u = Math.max(rng(), 1e-9)
  const v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

// Noisy-moons classification. Labels are 0/1, points live in [-1.8, 1.8]².
function makeData(n = BATCH): Array<{ x: number; y: number; label: 0 | 1 }> {
  const rng = mulberry32(11)
  const out: Array<{ x: number; y: number; label: 0 | 1 }> = []
  for (let i = 0; i < n; i++) {
    const t = (i / n) * Math.PI
    const nx = gauss(rng) * 0.1
    const ny = gauss(rng) * 0.1
    if (i % 2 === 0) {
      out.push({ x: Math.cos(t) - 0.5 + nx, y: Math.sin(t) + ny, label: 0 })
    } else {
      out.push({ x: Math.cos(t) + 0.5 + nx, y: -Math.sin(t) + 0.3 + ny, label: 1 })
    }
  }
  return out
}

interface MLP {
  W1: number[][] // (H, 2)
  b1: number[] // (H,)
  W2: number[] // (H,)
  b2: number
}

function initMLP(H: number, scale: number, rng: () => number): MLP {
  // scale is a multiplier on top of He: scale=1 is standard He init,
  // scale<1 shrinks weights (more dead neurons as signal dies),
  // scale>1 blows them up (also kills neurons once biases drift).
  const W1: number[][] = []
  for (let i = 0; i < H; i++) {
    const s = scale * Math.sqrt(2 / 2)
    W1.push([gauss(rng) * s, gauss(rng) * s])
  }
  const b1 = new Array(H).fill(0).map(() => gauss(rng) * 0.05 * scale)
  const W2 = new Array(H).fill(0).map(() => gauss(rng) * Math.sqrt(2 / H))
  return { W1, b1, W2, b2: 0 }
}

function relu(z: number): number {
  return Math.max(0, z)
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z))
}

interface Forward {
  z1: number[][]
  a1: number[][]
  yhat: number[]
}

function forwardBatch(net: MLP, data: Array<{ x: number; y: number }>): Forward {
  const N = data.length
  const H = net.W1.length
  const z1: number[][] = []
  const a1: number[][] = []
  const yhat: number[] = []
  for (let n = 0; n < N; n++) {
    const row: number[] = []
    const act: number[] = []
    for (let h = 0; h < H; h++) {
      const z = net.W1[h][0] * data[n].x + net.W1[h][1] * data[n].y + net.b1[h]
      row.push(z)
      act.push(relu(z))
    }
    z1.push(row)
    a1.push(act)
    let z2 = net.b2
    for (let h = 0; h < H; h++) z2 += net.W2[h] * act[h]
    yhat.push(sigmoid(z2))
  }
  return { z1, a1, yhat }
}

function trainStep(
  net: MLP,
  data: Array<{ x: number; y: number; label: 0 | 1 }>,
  lr: number,
): { loss: number; rates: number[] } {
  const H = net.W1.length
  const { z1, a1, yhat } = forwardBatch(net, data)
  const N = data.length

  let loss = 0
  const dW1 = net.W1.map((r) => r.map(() => 0))
  const db1 = new Array(H).fill(0)
  const dW2 = new Array(H).fill(0)
  let db2 = 0
  const active = new Array(H).fill(0)

  for (let n = 0; n < N; n++) {
    const y = data[n].label
    const p = yhat[n]
    loss += -(y * Math.log(Math.max(p, 1e-9)) + (1 - y) * Math.log(Math.max(1 - p, 1e-9)))
    const delta2 = (p - y) / N
    db2 += delta2
    for (let h = 0; h < H; h++) {
      dW2[h] += delta2 * a1[n][h]
      const gate = z1[n][h] > 0 ? 1 : 0
      if (gate) active[h] += 1
      const d1h = net.W2[h] * delta2 * gate
      db1[h] += d1h
      dW1[h][0] += d1h * data[n].x
      dW1[h][1] += d1h * data[n].y
    }
  }

  for (let h = 0; h < H; h++) {
    net.W1[h][0] -= lr * dW1[h][0]
    net.W1[h][1] -= lr * dW1[h][1]
    net.b1[h] -= lr * db1[h]
    net.W2[h] -= lr * dW2[h]
  }
  net.b2 -= lr * db2

  return { loss: loss / N, rates: active.map((c) => c / N) }
}

export default function LiveNeuronMonitor() {
  const [scale, setScale] = useState(1.0)
  const [playing, setPlaying] = useState(false)
  const [step, setStep] = useState(0)
  const [loss, setLoss] = useState(0.69)
  const [rates, setRates] = useState<number[]>(() => new Array(HIDDEN).fill(0.5))
  const netRef = useRef<MLP | null>(null)
  const rafRef = useRef<number | null>(null)

  const data = useMemo(() => makeData(), [])

  const resetNet = (s = scale) => {
    const rng = mulberry32((Date.now() & 0xffff) + Math.floor(s * 1000))
    netRef.current = initMLP(HIDDEN, s, rng)
    setStep(0)
    setLoss(0.69)
    // Seed the initial rates from a no-train forward pass.
    const { z1 } = forwardBatch(netRef.current, data)
    const init = new Array(HIDDEN).fill(0)
    for (const row of z1) {
      for (let h = 0; h < HIDDEN; h++) if (row[h] > 0) init[h] += 1
    }
    setRates(init.map((c) => c / data.length))
  }

  // Rebuild whenever scale changes — the whole point is to watch cells go dark.
  useEffect(() => {
    resetNet(scale)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scale])

  useEffect(() => {
    if (!playing || !netRef.current) return
    const tick = () => {
      if (!netRef.current) return
      let L = loss
      let R: number[] = rates
      for (let i = 0; i < 2; i++) {
        const { loss: l, rates: r } = trainStep(netRef.current!, data, 0.25)
        L = l
        R = r
      }
      setLoss(L)
      setRates(R)
      setStep((s) => s + 2)
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing])

  // Classification: a neuron is *dead* if its activation rate is exactly 0
  // (every example in the batch pushed it to the negative side). It is
  // *asleep* if the rate is tiny — positive but under 5% — typically a
  // neuron that only fires on a handful of outliers. Otherwise alive.
  const dead = rates.filter((r) => r === 0).length
  const asleep = rates.filter((r) => r > 0 && r < 0.05).length
  const alive = HIDDEN - dead - asleep

  return (
    <WidgetFrame
      widgetName="LiveNeuronMonitor"
      label="live neuron monitor — 32 ReLU cells, activation rate per batch"
      right={
        <>
          <span className="font-mono">2 → 32 (ReLU) → 1</span>
          <span className="text-dark-text-disabled">·</span>
          <span className="font-mono">batch {BATCH}</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="init scale"
            value={scale}
            min={0.05}
            max={4.0}
            step={0.01}
            onChange={setScale}
            format={(v) => `${v.toFixed(2)}×`}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-2">
            <Button onClick={() => setPlaying((p) => !p)} variant="primary">
              {playing ? (
                <>
                  <Pause className="w-3 h-3 inline -mt-px mr-1" /> pause
                </>
              ) : (
                <>
                  <Play className="w-3 h-3 inline -mt-px mr-1" /> train
                </>
              )}
            </Button>
            <Button
              onClick={() => {
                setPlaying(false)
                resetNet()
              }}
            >
              <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
            </Button>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="step" value={String(step)} />
            <Readout label="loss" value={loss.toFixed(3)} accent="text-term-amber" />
            <Readout label="alive" value={String(alive)} accent="text-term-green" />
            <Readout
              label="asleep"
              value={String(asleep)}
              accent={asleep > 0 ? 'text-term-amber' : 'text-dark-text-disabled'}
            />
            <Readout
              label="dead"
              value={String(dead)}
              accent={dead > 0 ? 'text-term-rose' : 'text-dark-text-disabled'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 flex items-center justify-center p-6 gap-6">
        <div
          className="grid gap-2 w-full h-full max-w-[520px] max-h-[300px]"
          style={{ gridTemplateColumns: `repeat(${GRID_COLS}, 1fr)` }}
        >
          {rates.map((r, i) => {
            let bg: string
            let border: string
            let label: string
            if (r === 0) {
              bg = 'rgba(10, 10, 10, 1)'
              border = 'rgba(244, 63, 94, 0.55)'
              label = 'dead'
            } else if (r < 0.05) {
              bg = `rgba(251, 191, 36, ${0.08 + r * 0.6})`
              border = 'rgba(251, 191, 36, 0.35)'
              label = 'asleep'
            } else {
              const a = 0.2 + r * 0.8
              bg = `rgba(251, 191, 36, ${a})`
              border = `rgba(251, 191, 36, ${Math.min(1, a + 0.2)})`
              label = 'active'
            }
            return (
              <div
                key={i}
                className="rounded-md transition-all flex items-end justify-end p-1"
                style={{
                  backgroundColor: bg,
                  boxShadow: `inset 0 0 0 1px ${border}`,
                }}
                title={`neuron #${i} — ${label}, rate ${(r * 100).toFixed(0)}%`}
              >
                <span className="text-[9px] font-mono text-dark-bg/70 tabular-nums">
                  {(r * 100).toFixed(0)}
                </span>
              </div>
            )
          })}
        </div>
      </div>
      <div className="absolute bottom-2 left-4 right-4 text-[11px] font-mono text-dark-text-muted pointer-events-none flex items-center gap-4">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 rounded-sm bg-term-amber/80" />
          active (rate &gt; 5%)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 rounded-sm bg-term-amber/20 border border-term-amber/40" />
          asleep (0 &lt; rate &lt; 5%)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 rounded-sm bg-dark-bg border border-term-rose/50" />
          dead (rate = 0, never updates)
        </span>
      </div>
    </WidgetFrame>
  )
}
