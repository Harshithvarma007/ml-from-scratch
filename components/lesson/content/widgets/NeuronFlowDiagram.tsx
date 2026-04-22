'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// A proper wiring diagram. Three inputs feed into a neuron body via weighted
// edges. The edge thickness/color scales with |wᵢ·xᵢ| so the "signals" flowing
// in are visually proportional to their contribution. Click an input to
// negate its weight — watch the output flip in real time.

type ActName = 'relu' | 'sigmoid' | 'tanh'

function act(a: ActName, z: number): number {
  if (a === 'relu') return Math.max(0, z)
  if (a === 'sigmoid') return 1 / (1 + Math.exp(-z))
  return Math.tanh(z)
}

const inputLabels = ['price', 'reviews', 'rating']
const defaultInputs = [0.8, 0.6, 0.9]

export default function NeuronFlowDiagram() {
  const [weights, setWeights] = useState([0.5, 0.3, 0.8])
  const [inputs] = useState(defaultInputs)
  const [bias, setBias] = useState(-0.4)
  const [activation, setActivation] = useState<ActName>('sigmoid')

  const products = useMemo(() => inputs.map((x, i) => x * weights[i]), [inputs, weights])
  const z = products.reduce((s, p) => s + p, 0) + bias
  const y = act(activation, z)

  const maxAbs = Math.max(...products.map(Math.abs), 0.2)

  const flipWeight = (i: number) => {
    const next = [...weights]
    next[i] = -next[i]
    setWeights(next)
  }

  const nudgeWeight = (i: number, d: number) => {
    const next = [...weights]
    next[i] = Math.max(-1.2, Math.min(1.2, next[i] + d))
    setWeights(next)
  }

  return (
    <WidgetFrame
      widgetName="NeuronFlowDiagram"
      label="the signal flow — why 'synapse' is the right metaphor"
      right={<span className="font-mono">edge thickness ∝ |wᵢ · xᵢ|</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            <span className="text-[11px] font-mono text-dark-text-disabled uppercase tracking-wider mr-1">
              activation
            </span>
            {(['relu', 'sigmoid', 'tanh'] as ActName[]).map((a) => (
              <button
                key={a}
                onClick={() => setActivation(a)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
                  activation === a
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
                )}
              >
                {a}
              </button>
            ))}
          </div>
          <button
            onClick={() => setBias((b) => Math.max(-1.2, b - 0.1))}
            className="px-2 py-1 text-[11px] font-mono border border-dark-border rounded text-dark-text-secondary hover:text-dark-text-primary"
          >
            b −
          </button>
          <span className="text-[11px] font-mono text-dark-text-muted tabular-nums">
            b = {bias.toFixed(2)}
          </span>
          <button
            onClick={() => setBias((b) => Math.min(1.2, b + 0.1))}
            className="px-2 py-1 text-[11px] font-mono border border-dark-border rounded text-dark-text-secondary hover:text-dark-text-primary"
          >
            b +
          </button>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="pre-act z" value={z.toFixed(3)} />
            <Readout label="output y" value={y.toFixed(4)} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 flex items-center justify-center p-6">
        <svg viewBox="0 0 800 360" className="w-full h-full">
          {/* Input column */}
          {inputs.map((x, i) => {
            const cy = 80 + i * 100
            return (
              <g key={i}>
                <circle cx={120} cy={cy} r={28} fill="#1f1b3a" stroke="#a78bfa" strokeWidth={1.5} />
                <text x={120} y={cy - 2} textAnchor="middle" fontSize="10" fill="#888" fontFamily="JetBrains Mono, monospace">
                  {inputLabels[i]}
                </text>
                <text x={120} y={cy + 14} textAnchor="middle" fontSize="13" fill="#fbbf24" fontFamily="JetBrains Mono, monospace" fontWeight="600">
                  {x.toFixed(2)}
                </text>
              </g>
            )
          })}

          {/* Edges */}
          {inputs.map((x, i) => {
            const y1 = 80 + i * 100
            const strokeW = 1 + (Math.abs(products[i]) / maxAbs) * 5
            const color = products[i] >= 0 ? '#67e8f9' : '#f472b6'
            const midX = 300
            const midY = y1 + (180 - y1) * 0.5
            return (
              <g key={i}>
                <path
                  d={`M ${148} ${y1} Q ${midX} ${midY}, ${472} 180`}
                  fill="none"
                  stroke={color}
                  strokeOpacity={0.7}
                  strokeWidth={strokeW}
                />
                <g
                  onClick={() => flipWeight(i)}
                  style={{ cursor: 'pointer' }}
                >
                  <rect
                    x={midX - 40}
                    y={midY - 14}
                    width={80}
                    height={26}
                    rx={4}
                    fill="#0a0a0a"
                    stroke={color}
                    strokeWidth={1}
                    opacity={0.92}
                  />
                  <text
                    x={midX}
                    y={midY - 2}
                    textAnchor="middle"
                    fontSize="10"
                    fill="#888"
                    fontFamily="JetBrains Mono, monospace"
                  >
                    w{sub(i + 1)} = {weights[i].toFixed(2)}
                  </text>
                  <text
                    x={midX}
                    y={midY + 8}
                    textAnchor="middle"
                    fontSize="9"
                    fill={color}
                    fontFamily="JetBrains Mono, monospace"
                  >
                    x·w = {products[i].toFixed(2)}
                  </text>
                </g>
                {/* nudge arrows */}
                <g>
                  <text
                    x={midX - 60}
                    y={midY + 4}
                    fontSize="14"
                    fill="#555"
                    fontFamily="JetBrains Mono, monospace"
                    onClick={() => nudgeWeight(i, -0.1)}
                    style={{ cursor: 'pointer' }}
                  >
                    −
                  </text>
                  <text
                    x={midX + 52}
                    y={midY + 4}
                    fontSize="14"
                    fill="#555"
                    fontFamily="JetBrains Mono, monospace"
                    onClick={() => nudgeWeight(i, 0.1)}
                    style={{ cursor: 'pointer' }}
                  >
                    +
                  </text>
                </g>
              </g>
            )
          })}

          {/* Neuron body */}
          <circle cx={500} cy={180} r={44} fill="#1a1a1a" stroke="#a78bfa" strokeWidth={2} />
          <text x={500} y={172} textAnchor="middle" fontSize="10" fill="#888" fontFamily="JetBrains Mono, monospace">
            Σ + b = {z.toFixed(2)}
          </text>
          <text x={500} y={188} textAnchor="middle" fontSize="10" fill="#888" fontFamily="JetBrains Mono, monospace">
            ↓ {activation}
          </text>
          <text
            x={500}
            y={204}
            textAnchor="middle"
            fontSize="13"
            fill="#fbbf24"
            fontFamily="JetBrains Mono, monospace"
            fontWeight="700"
          >
            {y.toFixed(3)}
          </text>

          {/* Output arrow */}
          <path
            d={`M 544 180 L 660 180`}
            stroke="#fbbf24"
            strokeWidth={2 + y * 4}
            strokeOpacity={0.85}
            fill="none"
            markerEnd="url(#arrow)"
          />
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="10"
              refY="5"
              markerWidth="8"
              markerHeight="8"
              orient="auto"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#fbbf24" />
            </marker>
          </defs>
          <text x={700} y={178} textAnchor="middle" fontSize="10" fill="#888" fontFamily="JetBrains Mono, monospace">
            output
          </text>
          <text x={700} y={196} textAnchor="middle" fontSize="13" fill="#fbbf24" fontFamily="JetBrains Mono, monospace" fontWeight="700">
            {y.toFixed(3)}
          </text>
        </svg>
      </div>
      <div className="absolute bottom-2 left-4 text-[10.5px] font-mono text-dark-text-disabled pointer-events-none">
        click an edge pill to flip the sign of its weight · hit +/− to nudge
      </div>
    </WidgetFrame>
  )
}

function sub(n: number): string {
  return String(n)
    .split('')
    .map((d) => '₀₁₂₃₄₅₆₇₈₉'[Number(d)] ?? d)
    .join('')
}
