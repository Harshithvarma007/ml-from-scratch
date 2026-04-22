'use client'

import { useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Residual block SVG diagram with toggle for skip. Compute gradient at input
// assuming gradient=1 at output. Without skip: prod of 3 local derivatives.
// With skip: 1 + prod. Show how the +1 rescues vanishing.

export default function SkipConnectionViz() {
  const [skip, setSkip] = useState(true)
  const [depth, setDepth] = useState(10)
  const [d, setD] = useState(0.5) // average local derivative magnitude

  const plainGrad = Math.pow(d, depth)
  const residualGrad = Math.pow(1 + d, depth) // simplified: gradient through L residual blocks

  return (
    <WidgetFrame
      widgetName="SkipConnectionViz"
      label="skip connection — the identity path that kept gradients alive"
      right={<span className="font-mono">y = F(x) + x  ·  ∂y/∂x = 1 + ∂F/∂x</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <button
            onClick={() => setSkip((v) => !v)}
            className={cn(
              'px-3 py-1 rounded text-[11px] font-mono uppercase tracking-wider transition-all',
              skip
                ? 'bg-term-green text-dark-bg'
                : 'border border-dark-border text-dark-text-secondary',
            )}
          >
            skip: {skip ? 'on' : 'off'}
          </button>
          <Slider
            label="depth L"
            value={depth}
            min={1}
            max={40}
            step={1}
            onChange={(v) => setDepth(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-purple"
          />
          <Slider
            label="local |∂F/∂x|"
            value={d}
            min={0.1}
            max={1.5}
            step={0.05}
            onChange={setD}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="plain grad"
              value={fmt(plainGrad)}
              accent={plainGrad < 1e-3 ? 'text-term-rose' : 'text-term-amber'}
            />
            <Readout
              label="residual grad"
              value={fmt(residualGrad)}
              accent="text-term-green"
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 flex items-center justify-center">
        <svg viewBox="0 0 800 360" className="w-full h-full max-w-[900px]">
          {/* Input */}
          <g>
            <circle cx={70} cy={180} r={28} fill="#1f1b3a" stroke="#67e8f9" strokeWidth={1.5} />
            <text x={70} y={184} textAnchor="middle" fontSize="13" fill="#67e8f9" fontFamily="JetBrains Mono, monospace">
              x
            </text>
          </g>

          {/* Skip arc */}
          {skip && (
            <>
              <path
                d="M 90 170 Q 400 40 710 170"
                stroke="#4ade80"
                strokeWidth={3}
                fill="none"
                strokeOpacity={0.8}
              />
              <text x={400} y={55} textAnchor="middle" fontSize="11" fill="#4ade80" fontFamily="JetBrains Mono, monospace">
                identity · gradient = 1
              </text>
            </>
          )}

          {/* F(x) path: three operations */}
          <path d="M 100 180 L 180 180" stroke="#a78bfa" strokeWidth={2} />
          <g>
            <rect x={180} y={155} width={120} height={50} rx={6} fill="#1a1a1a" stroke="#a78bfa" strokeWidth={1.5} />
            <text x={240} y={176} textAnchor="middle" fontSize="11" fill="#a78bfa" fontFamily="JetBrains Mono, monospace" fontWeight="600">
              Conv 3×3
            </text>
            <text x={240} y={192} textAnchor="middle" fontSize="10" fill="#888" fontFamily="JetBrains Mono, monospace">
              ∂ = {d.toFixed(2)}
            </text>
          </g>
          <path d="M 300 180 L 340 180" stroke="#a78bfa" strokeWidth={2} />
          <g>
            <rect x={340} y={155} width={120} height={50} rx={6} fill="#1a1a1a" stroke="#a78bfa" strokeWidth={1.5} />
            <text x={400} y={176} textAnchor="middle" fontSize="11" fill="#a78bfa" fontFamily="JetBrains Mono, monospace" fontWeight="600">
              BatchNorm + ReLU
            </text>
            <text x={400} y={192} textAnchor="middle" fontSize="10" fill="#888" fontFamily="JetBrains Mono, monospace">
              ∂ = {d.toFixed(2)}
            </text>
          </g>
          <path d="M 460 180 L 500 180" stroke="#a78bfa" strokeWidth={2} />
          <g>
            <rect x={500} y={155} width={120} height={50} rx={6} fill="#1a1a1a" stroke="#a78bfa" strokeWidth={1.5} />
            <text x={560} y={176} textAnchor="middle" fontSize="11" fill="#a78bfa" fontFamily="JetBrains Mono, monospace" fontWeight="600">
              Conv 3×3
            </text>
            <text x={560} y={192} textAnchor="middle" fontSize="10" fill="#888" fontFamily="JetBrains Mono, monospace">
              ∂ = {d.toFixed(2)}
            </text>
          </g>
          <path d="M 620 180 L 680 180" stroke="#a78bfa" strokeWidth={2} />

          {/* Sum node */}
          <g>
            <circle cx={710} cy={180} r={24} fill="#1a1a1a" stroke={skip ? '#4ade80' : '#a78bfa'} strokeWidth={2} />
            <text x={710} y={185} textAnchor="middle" fontSize="16" fill={skip ? '#4ade80' : '#a78bfa'} fontFamily="JetBrains Mono, monospace" fontWeight="700">
              {skip ? '+' : '⋅'}
            </text>
          </g>

          {/* Output */}
          <path d="M 734 180 L 770 180" stroke="#fbbf24" strokeWidth={2} />
          <text x={760} y={200} fontSize="11" fill="#fbbf24" fontFamily="JetBrains Mono, monospace">
            y
          </text>

          {/* Gradient annotations */}
          <g>
            <text x={400} y={280} textAnchor="middle" fontSize="12" fill="#888" fontFamily="JetBrains Mono, monospace">
              {skip
                ? `∂y/∂x = 1 + (${d.toFixed(2)})³ ≈ ${(1 + Math.pow(d, 3)).toFixed(3)}`
                : `∂y/∂x = (${d.toFixed(2)})³ = ${Math.pow(d, 3).toFixed(3)}`}
            </text>
            <text x={400} y={305} textAnchor="middle" fontSize="11" fill={skip ? '#4ade80' : '#f87171'} fontFamily="JetBrains Mono, monospace">
              over L={depth} blocks: {skip ? `(1 + ${d.toFixed(2)})^${depth}` : `(${d.toFixed(2)})^${depth}`} ≈ {fmt(skip ? residualGrad : plainGrad)}
            </text>
          </g>
        </svg>
      </div>
    </WidgetFrame>
  )
}

function fmt(v: number): string {
  if (v === 0) return '0'
  if (v < 1e-3 || v > 1e4) return v.toExponential(1)
  return v.toFixed(3)
}
