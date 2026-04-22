'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// A deliberately finger-pointable MLP. Adjust input dim, depth, width, output
// dim — see a scaled schematic of the network. Parameter count + FLOP count
// update live. Teaches the intuition "depth × width² + in·H + H·out" without
// hand-waving.

export default function MLPArchitecture() {
  const [inputDim, setInputDim] = useState(10)
  const [depth, setDepth] = useState(3) // number of hidden layers
  const [width, setWidth] = useState(64)
  const [outputDim, setOutputDim] = useState(1)

  const layerDims = useMemo(() => {
    const arr = [inputDim]
    for (let i = 0; i < depth; i++) arr.push(width)
    arr.push(outputDim)
    return arr
  }, [inputDim, depth, width, outputDim])

  const params = useMemo(() => {
    let p = 0
    for (let i = 0; i < layerDims.length - 1; i++) {
      p += layerDims[i] * layerDims[i + 1] + layerDims[i + 1]
    }
    return p
  }, [layerDims])

  // FLOPs per forward pass (multiply–add is 2 ops).
  const flops = useMemo(() => {
    let f = 0
    for (let i = 0; i < layerDims.length - 1; i++) {
      f += 2 * layerDims[i] * layerDims[i + 1]
    }
    return f
  }, [layerDims])

  return (
    <WidgetFrame
      widgetName="MLPArchitecture"
      label="architecture — params and FLOPs you can feel"
      right={
        <>
          <span className="font-mono">params = Σ (inᵢ · outᵢ + outᵢ)</span>
          <span className="text-dark-text-disabled">·</span>
          <span className="font-mono">FLOPs ≈ 2 · Σ inᵢ · outᵢ</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="input"
            value={inputDim}
            min={1}
            max={256}
            step={1}
            onChange={(v) => setInputDim(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-cyan"
          />
          <Slider
            label="depth"
            value={depth}
            min={1}
            max={8}
            step={1}
            onChange={(v) => setDepth(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-dark-accent"
          />
          <Slider
            label="width"
            value={width}
            min={4}
            max={512}
            step={1}
            onChange={(v) => setWidth(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-purple"
          />
          <Slider
            label="output"
            value={outputDim}
            min={1}
            max={100}
            step={1}
            onChange={(v) => setOutputDim(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="params" value={fmt(params)} accent="text-term-amber" />
            <Readout label="FLOPs/fwd" value={fmt(flops)} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-6 flex items-center justify-center overflow-auto">
        <svg viewBox={`0 0 860 340`} className="w-full h-full max-w-[860px]">
          {layerDims.map((dim, layer) => {
            const x = 60 + (layer / (layerDims.length - 1)) * (860 - 120)
            // Cap visible circles at 12 per layer; otherwise dot-dot-dot.
            const visible = Math.min(dim, 12)
            const spacing = 20
            const totalH = (visible - 1) * spacing
            const startY = 170 - totalH / 2
            return (
              <g key={layer}>
                {Array.from({ length: visible }).map((_, i) => (
                  <circle
                    key={i}
                    cx={x}
                    cy={startY + i * spacing}
                    r={5}
                    fill={layer === 0 ? '#67e8f9' : layer === layerDims.length - 1 ? '#fbbf24' : '#a78bfa'}
                    opacity={0.8}
                  />
                ))}
                {dim > visible && (
                  <text
                    x={x}
                    y={startY + visible * spacing + 8}
                    textAnchor="middle"
                    fontSize="9"
                    fill="#666"
                    fontFamily="JetBrains Mono, monospace"
                  >
                    ⋮ {dim}
                  </text>
                )}
                <text
                  x={x}
                  y={45}
                  textAnchor="middle"
                  fontSize="10"
                  fill="#888"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {layer === 0
                    ? 'input'
                    : layer === layerDims.length - 1
                      ? 'output'
                      : `h${layer}`}
                </text>
                <text
                  x={x}
                  y={58}
                  textAnchor="middle"
                  fontSize="9"
                  fill="#555"
                  fontFamily="JetBrains Mono, monospace"
                >
                  dim={dim}
                </text>
              </g>
            )
          })}

          {/* Edges between successive layers — sparse sampling */}
          {layerDims.slice(0, -1).map((inD, l) => {
            const outD = layerDims[l + 1]
            const x1 = 60 + (l / (layerDims.length - 1)) * (860 - 120)
            const x2 = 60 + ((l + 1) / (layerDims.length - 1)) * (860 - 120)
            const n1 = Math.min(inD, 12)
            const n2 = Math.min(outD, 12)
            const y1s = Array.from({ length: n1 }, (_, i) => 170 - (n1 - 1) * 10 + i * 20)
            const y2s = Array.from({ length: n2 }, (_, i) => 170 - (n2 - 1) * 10 + i * 20)
            const edges: Array<[number, number, number, number]> = []
            for (const a of y1s) for (const b of y2s) edges.push([x1, a, x2, b])
            return (
              <g key={l}>
                {edges.map(([x1, y1, x2, y2], i) => (
                  <line
                    key={i}
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="#a78bfa"
                    strokeOpacity={0.08}
                    strokeWidth={0.5}
                  />
                ))}
              </g>
            )
          })}

          {/* Per-layer param count labels */}
          {layerDims.slice(0, -1).map((inD, l) => {
            const outD = layerDims[l + 1]
            const x = 60 + ((l + 0.5) / (layerDims.length - 1)) * (860 - 120)
            const p = inD * outD + outD
            return (
              <text
                key={l}
                x={x}
                y={310}
                textAnchor="middle"
                fontSize="10"
                fill="#666"
                fontFamily="JetBrains Mono, monospace"
              >
                {fmt(p)} params
              </text>
            )
          })}
        </svg>
      </div>
    </WidgetFrame>
  )
}

function fmt(n: number): string {
  if (n < 1000) return String(n)
  if (n < 1e6) return (n / 1e3).toFixed(1) + 'K'
  if (n < 1e9) return (n / 1e6).toFixed(1) + 'M'
  return (n / 1e9).toFixed(1) + 'B'
}
