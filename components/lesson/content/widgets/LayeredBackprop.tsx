'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Button } from './WidgetFrame'
import { Play, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'

// A three-layer fully connected network drawn as circles + edges. Step through
// the forward pass (left→right), then the backward pass (right→left). Edge
// colour encodes the value or gradient being passed along that edge. Makes
// "δ flows backward" into something you can watch.

// Hand-crafted small network: 3 → 4 → 3 → 1.
const LAYERS = [3, 4, 3, 1]

// Deterministic pseudo-random weights so the animation is reproducible.
function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

interface Network {
  W: number[][][] // W[layer][out][in]
  b: number[][] // b[layer][out]
  X: number[]
  target: number
}

function buildNet(): Network {
  const rng = mulberry32(5)
  const W: number[][][] = []
  const b: number[][] = []
  for (let l = 0; l < LAYERS.length - 1; l++) {
    const inD = LAYERS[l]
    const outD = LAYERS[l + 1]
    const Wl: number[][] = []
    for (let i = 0; i < outD; i++) {
      const row: number[] = []
      for (let j = 0; j < inD; j++) row.push(rng() * 1.4 - 0.7)
      Wl.push(row)
    }
    W.push(Wl)
    b.push(new Array(outD).fill(0).map(() => rng() * 0.4 - 0.2))
  }
  return { W, b, X: [0.6, -0.4, 0.9], target: 0.8 }
}

function relu(z: number): number {
  return Math.max(0, z)
}

function forward(net: Network): { acts: number[][]; pre: number[][]; loss: number } {
  const acts: number[][] = [net.X]
  const pre: number[][] = [net.X]
  for (let l = 0; l < net.W.length; l++) {
    const input = acts[l]
    const preOut: number[] = []
    const out: number[] = []
    for (let i = 0; i < net.W[l].length; i++) {
      let z = net.b[l][i]
      for (let j = 0; j < input.length; j++) z += net.W[l][i][j] * input[j]
      preOut.push(z)
      // ReLU on hidden layers, linear on the output
      out.push(l === net.W.length - 1 ? z : relu(z))
    }
    pre.push(preOut)
    acts.push(out)
  }
  const yhat = acts[acts.length - 1][0]
  return { acts, pre, loss: 0.5 * (yhat - net.target) ** 2 }
}

function backward(
  net: Network,
  acts: number[][],
  pre: number[][],
): { deltas: number[][] } {
  // δ for the output layer: (yhat - y) * 1 (linear)
  const L = net.W.length
  const deltas: number[][] = new Array(L + 1)
  deltas[L] = [acts[L][0] - net.target]
  for (let l = L - 1; l >= 1; l--) {
    const upstream = deltas[l + 1]
    const W = net.W[l] // (outD, inD)  upstream has outD entries
    const delta: number[] = []
    for (let j = 0; j < acts[l].length; j++) {
      let s = 0
      for (let i = 0; i < W.length; i++) s += upstream[i] * W[i][j]
      // ReLU'(pre[l][j])
      const relug = pre[l][j] > 0 ? 1 : 0
      delta.push(s * relug)
    }
    deltas[l] = delta
  }
  deltas[0] = [] // inputs have no δ
  return { deltas }
}

const STEPS = 2 * LAYERS.length - 1 // forward through each layer, then back through each

export default function LayeredBackprop() {
  const [step, setStep] = useState(0)
  const net = useMemo(() => buildNet(), [])
  const { acts, pre, loss } = useMemo(() => forward(net), [net])
  const { deltas } = useMemo(() => backward(net, acts, pre), [net, acts, pre])

  const forwardDone = Math.min(step, LAYERS.length - 1) // layers 0..forwardDone have values
  const backwardDone = Math.max(0, step - LAYERS.length + 1) // gradients flow back this many steps

  const phase =
    step < LAYERS.length
      ? step === 0
        ? 'waiting'
        : 'forward pass'
      : step >= STEPS
        ? 'complete'
        : 'backward pass'
  const phaseColor = step < LAYERS.length ? 'text-term-cyan' : 'text-term-rose'

  // Layout — evenly space layers horizontally
  const viewW = 800
  const viewH = 340
  const layerX = (i: number) => 80 + (i * (viewW - 160)) / (LAYERS.length - 1)
  const nodeY = (layer: number, idx: number) => {
    const n = LAYERS[layer]
    const spacing = 55
    const total = (n - 1) * spacing
    return viewH / 2 - total / 2 + idx * spacing
  }

  return (
    <WidgetFrame
      widgetName="LayeredBackprop"
      label="multi-layer backprop — δ flows right to left"
      right={
        <>
          <span className="font-mono">3 → 4 → 3 → 1, ReLU hidden, linear out</span>
          <span className="text-dark-text-disabled">·</span>
          <span className={cn('font-mono', phaseColor)}>{phase}</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Button
            onClick={() => setStep((s) => Math.min(STEPS, s + 1))}
            variant="primary"
            disabled={step >= STEPS}
          >
            <Play className="w-3 h-3 inline -mt-px mr-1" /> next step
          </Button>
          <Button onClick={() => setStep(0)}>
            <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
          </Button>
          <span className="text-[11px] font-mono text-dark-text-muted">
            step {step} / {STEPS}
          </span>
          <div className="flex items-center gap-4 ml-auto text-[11px] font-mono">
            <span className="text-dark-text-muted">loss = {loss.toFixed(4)}</span>
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4">
        <svg viewBox={`0 0 ${viewW} ${viewH}`} className="w-full h-full">
          {/* Forward edges */}
          {net.W.map((Wl, l) => {
            const inLayer = l
            const outLayer = l + 1
            const reveal = inLayer < forwardDone
            return Wl.flatMap((row, i) =>
              row.map((w, j) => {
                const x1 = layerX(inLayer)
                const y1 = nodeY(inLayer, j)
                const x2 = layerX(outLayer)
                const y2 = nodeY(outLayer, i)
                // Show gradient flow during backward — reverse direction highlighting
                const backwardReveal =
                  step >= LAYERS.length &&
                  outLayer >= LAYERS.length - backwardDone
                const color = backwardReveal
                  ? w * (deltas[outLayer][i] ?? 0) >= 0
                    ? '#f472b6'
                    : '#a78bfa'
                  : reveal
                    ? w >= 0
                      ? '#67e8f9'
                      : '#f472b6'
                    : '#2a2a2a'
                const opacity = reveal || backwardReveal ? 0.7 : 0.3
                const width = 0.8 + Math.abs(w) * 1.2
                return (
                  <line
                    key={`${l}-${i}-${j}`}
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke={color}
                    strokeWidth={width}
                    strokeOpacity={opacity}
                  />
                )
              }),
            )
          })}

          {/* Nodes */}
          {LAYERS.map((n, layer) =>
            Array.from({ length: n }).map((_, i) => {
              const x = layerX(layer)
              const y = nodeY(layer, i)
              const hasForward = layer <= forwardDone
              const hasBackward =
                step >= LAYERS.length &&
                layer >= LAYERS.length - 1 - backwardDone
              const value = hasForward ? acts[layer][i].toFixed(2) : ''
              const delta =
                hasBackward && layer > 0
                  ? deltas[layer]?.[i]?.toFixed(3)
                  : undefined
              return (
                <g key={`n-${layer}-${i}`}>
                  <circle
                    cx={x}
                    cy={y}
                    r={20}
                    fill="#1a1a1a"
                    stroke={hasForward ? '#a78bfa' : '#444'}
                    strokeWidth={1.5}
                  />
                  {/* Forward value */}
                  {hasForward && (
                    <text
                      x={x}
                      y={y + 4}
                      textAnchor="middle"
                      fontSize="10"
                      fill="#fbbf24"
                      fontFamily="JetBrains Mono, monospace"
                    >
                      {value}
                    </text>
                  )}
                  {/* Backward δ bubble to the right of the node */}
                  {delta !== undefined && (
                    <g>
                      <rect
                        x={x + 22}
                        y={y - 10}
                        width={56}
                        height={18}
                        rx={4}
                        fill="#0a0a0a"
                        stroke="#f472b6"
                        strokeWidth={1}
                        opacity={0.9}
                      />
                      <text
                        x={x + 50}
                        y={y + 3}
                        textAnchor="middle"
                        fontSize="9"
                        fill="#f472b6"
                        fontFamily="JetBrains Mono, monospace"
                      >
                        δ={delta}
                      </text>
                    </g>
                  )}
                </g>
              )
            }),
          )}

          {/* Layer labels */}
          {LAYERS.map((n, layer) => (
            <text
              key={`lbl-${layer}`}
              x={layerX(layer)}
              y={viewH - 18}
              textAnchor="middle"
              fontSize="10"
              fill="#666"
              fontFamily="JetBrains Mono, monospace"
            >
              {layer === 0
                ? 'input'
                : layer === LAYERS.length - 1
                  ? 'output'
                  : `hidden ${layer}`}
            </text>
          ))}
        </svg>
      </div>

      <div className="absolute bottom-3 left-4 right-4 text-[10.5px] font-mono text-dark-text-disabled pointer-events-none flex items-center gap-6">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-0.5 bg-term-cyan" /> forward value flow
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-0.5 bg-term-pink" /> backward gradient flow
        </span>
        <span>Δ at each node = dL/dz for that neuron</span>
      </div>
    </WidgetFrame>
  )
}
