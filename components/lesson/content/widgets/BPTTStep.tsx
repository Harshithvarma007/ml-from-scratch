'use client'

import { useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { SkipBack, StepBack, StepForward } from 'lucide-react'

// Step through a BPTT backward pass on a tiny 5-step RNN. The forward pass
// runs once (shown as a tape along the top). The backward walk goes
// t = T, T-1, ..., each click fires one cell: pick up a local contribution
// to dW, multiply by W_h to hand dh_prev to the cell on the left. A panel
// below the highlighted cell shows the exact local math.

const T = 5
const W_H = 0.8     // stays in tanh's good zone so the signals are visible
const W_X = 0.7
const B = 0.0
const X = [0.8, 0.4, -0.3, 0.6, -0.9]
const TARGET = 0.5

const CELL_W = 90
const CELL_GAP = 70

// Cell t (1..T) has center x = CELL_X0 + (t-1) * (CELL_W + CELL_GAP)
const CELL_X0 = 95

function cellCX(t: number): number {
  return CELL_X0 + (t - 1) * (CELL_W + CELL_GAP)
}

function forward() {
  const h = [0]
  const z = [0]
  for (let t = 0; t < T; t++) {
    const zt = W_H * h[t] + W_X * X[t] + B
    z.push(zt)
    h.push(Math.tanh(zt))
  }
  return { h, z }
}

const { h, z } = forward()
const LOSS = 0.5 * (h[T] - TARGET) ** 2

// Each "state" the user can be in is characterized by `lastProcessed`:
// - null means nothing processed, fresh after reset
// - 5, 4, 3, 2, 1 mean that backward pass through cell t has completed
type State = {
  lastProcessed: number | null
  dh_in: number
  dz: number
  dW_contrib: number
  dh_out: number
  dW_total: number
}

function buildStates(): State[] {
  const out: State[] = []
  let dh = h[T] - TARGET
  let dW_total = 0
  // i = 0: nothing processed
  out.push({
    lastProcessed: null,
    dh_in: dh,
    dz: 0,
    dW_contrib: 0,
    dh_out: dh,
    dW_total: 0,
  })
  for (let t = T; t >= 1; t--) {
    const dtanh = 1 - Math.tanh(z[t]) ** 2
    const dz = dh * dtanh
    const dW_contrib = dz * h[t - 1]
    dW_total += dW_contrib
    const dh_prev = W_H * dz
    out.push({
      lastProcessed: t,
      dh_in: dh,
      dz,
      dW_contrib,
      dh_out: dh_prev,
      dW_total,
    })
    dh = dh_prev
  }
  return out
}

const STATES = buildStates()

export default function BPTTStep() {
  const [i, setI] = useState(0)
  const state = STATES[i]
  const { lastProcessed } = state

  // View-box is sized to fit T cells + loss + labels
  const LOSS_CX = cellCX(T) + CELL_W / 2 + CELL_GAP / 2 + 40
  const vbW = LOSS_CX + 120

  return (
    <WidgetFrame
      widgetName="BPTTStep"
      label="backprop through time — click step to walk backward"
      right={
        <span className="font-mono">
          L = ½(h_T − y)² = {LOSS.toFixed(4)} · target = {TARGET}
        </span>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <Button onClick={() => setI(0)} variant="ghost">
              <span className="inline-flex items-center gap-1">
                <SkipBack size={11} /> reset
              </span>
            </Button>
            <Button onClick={() => setI(Math.max(0, i - 1))} variant="ghost" disabled={i === 0}>
              <span className="inline-flex items-center gap-1">
                <StepBack size={11} /> back
              </span>
            </Button>
            <Button
              onClick={() => setI(Math.min(STATES.length - 1, i + 1))}
              variant="primary"
              disabled={i === STATES.length - 1}
            >
              <span className="inline-flex items-center gap-1">
                step <StepForward size={11} />
              </span>
            </Button>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="step" value={`${i} / ${STATES.length - 1}`} />
            <Readout label="∂L/∂h cursor" value={state.dh_in.toExponential(2)} accent="text-term-rose" />
            <Readout label="∂L/∂W_h (running)" value={state.dW_total.toExponential(2)} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-auto">
        <svg viewBox={`0 0 ${vbW} 340`} className="w-full h-full">
          {/* Forward tape */}
          {Array.from({ length: T }).map((_, k) => {
            const tNum = k + 1
            const cx = cellCX(tNum)
            const fired = lastProcessed !== null && lastProcessed <= tNum
            const isCurrent = lastProcessed === tNum
            return (
              <g key={k}>
                <rect
                  x={cx - CELL_W / 2}
                  y={60}
                  width={CELL_W}
                  height={60}
                  rx={8}
                  fill={isCurrent ? '#1f2937' : '#141420'}
                  stroke={isCurrent ? '#f87171' : fired ? '#7f1d1d' : '#3f3f46'}
                  strokeWidth={isCurrent ? 2 : 1.2}
                />
                <text x={cx} y={80} textAnchor="middle" fontSize="10" fill={isCurrent ? '#f87171' : '#a1a1aa'} fontFamily="JetBrains Mono, monospace">
                  t = {tNum}
                </text>
                <text x={cx} y={101} textAnchor="middle" fontSize="11" fill="#e5e7eb" fontFamily="JetBrains Mono, monospace">
                  h_{tNum} = {h[tNum].toFixed(2)}
                </text>
                <text x={cx} y={115} textAnchor="middle" fontSize="9" fill="#666" fontFamily="JetBrains Mono, monospace">
                  z = {z[tNum].toFixed(2)}
                </text>
                {/* Forward arrow to next cell */}
                {tNum < T && (
                  <line
                    x1={cx + CELL_W / 2}
                    y1={90}
                    x2={cx + CELL_W / 2 + CELL_GAP}
                    y2={90}
                    stroke="#2a2a32"
                    strokeWidth={1.2}
                  />
                )}
              </g>
            )
          })}

          {/* Forward arrow from cell T into loss */}
          <line
            x1={cellCX(T) + CELL_W / 2}
            y1={90}
            x2={LOSS_CX - 40}
            y2={90}
            stroke="#2a2a32"
            strokeWidth={1.2}
          />

          {/* Loss box */}
          <rect x={LOSS_CX - 40} y={60} width={80} height={60} rx={8} fill="#1a1530" stroke="#f87171" strokeWidth={1.2} />
          <text x={LOSS_CX} y={85} textAnchor="middle" fontSize="10" fill="#f87171" fontFamily="JetBrains Mono, monospace">
            loss
          </text>
          <text x={LOSS_CX} y={104} textAnchor="middle" fontSize="10" fill="#e5e7eb" fontFamily="JetBrains Mono, monospace">
            {LOSS.toFixed(3)}
          </text>

          {/* Backward arrows row (loss → T → T-1 → ... → 0) */}
          {/* Arrow from loss into cell T first */}
          {(() => {
            const fired = lastProcessed !== null
            const isCurrent = lastProcessed === T
            const color = isCurrent ? '#f87171' : fired ? '#7f1d1d' : '#2a2a32'
            const sw = isCurrent ? 2.6 : fired ? 1.6 : 1.2
            return (
              <line
                x1={LOSS_CX - 40}
                y1={170}
                x2={cellCX(T) + CELL_W / 2}
                y2={170}
                stroke={color}
                strokeWidth={sw}
                markerEnd={fired ? 'url(#arrow-rose)' : 'url(#arrow-gray)'}
              />
            )
          })()}
          {Array.from({ length: T - 1 }).map((_, k) => {
            // Arrow from cell (T - k) to cell (T - k - 1). The arrow represents
            // dh_prev coming out of cell (T - k).
            const tFrom = T - k
            const tTo = tFrom - 1
            const fired = lastProcessed !== null && lastProcessed <= tFrom
            const isCurrent = lastProcessed === tTo
            const cxFrom = cellCX(tFrom)
            const cxTo = cellCX(tTo)
            const color = isCurrent ? '#f87171' : fired ? '#7f1d1d' : '#2a2a32'
            const sw = isCurrent ? 2.6 : fired ? 1.6 : 1.2
            return (
              <g key={`arrow-${k}`}>
                <line
                  x1={cxFrom - CELL_W / 2}
                  y1={170}
                  x2={cxTo + CELL_W / 2}
                  y2={170}
                  stroke={color}
                  strokeWidth={sw}
                  markerEnd={fired ? 'url(#arrow-rose)' : 'url(#arrow-gray)'}
                />
                {isCurrent && (
                  <text
                    x={(cxFrom + cxTo) / 2}
                    y={160}
                    textAnchor="middle"
                    fontSize="10"
                    fill="#f87171"
                    fontFamily="JetBrains Mono, monospace"
                  >
                    dh ← W_h · dz
                  </text>
                )}
              </g>
            )
          })}
          <text x={30} y={174} fontSize="10" fill="#f87171" fontFamily="JetBrains Mono, monospace">
            ∂L/∂h
          </text>

          {/* Gradient math panel under the just-processed cell */}
          {lastProcessed !== null && (
            <g transform={`translate(${Math.min(Math.max(cellCX(lastProcessed) - 155, 20), vbW - 320)}, 210)`}>
              <rect x={0} y={0} width={310} height={110} rx={6} fill="#0f0f1a" stroke="#f87171" strokeWidth={1.2} />
              <text x={155} y={18} textAnchor="middle" fontSize="11" fill="#f87171" fontFamily="JetBrains Mono, monospace" fontWeight={600}>
                processed step t = {lastProcessed}
              </text>
              <text x={12} y={40} fontSize="10.5" fill="#e5e7eb" fontFamily="JetBrains Mono, monospace">
                dz = dh · tanh&apos;(z) = {state.dz.toExponential(2)}
              </text>
              <text x={12} y={57} fontSize="10.5" fill="#e5e7eb" fontFamily="JetBrains Mono, monospace">
                dW_contrib = dz · h_(t−1) = {state.dW_contrib.toExponential(2)}
              </text>
              <text x={12} y={74} fontSize="10.5" fill="#4ade80" fontFamily="JetBrains Mono, monospace">
                dW_h += dW_contrib  →  {state.dW_total.toExponential(2)}
              </text>
              <text x={12} y={91} fontSize="10.5" fill="#67e8f9" fontFamily="JetBrains Mono, monospace">
                dh_(t−1) = W_h · dz = {state.dh_out.toExponential(2)}
              </text>
            </g>
          )}

          {lastProcessed === null && (
            <g transform={`translate(${vbW / 2 - 150}, 220)`}>
              <rect x={0} y={0} width={300} height={80} rx={6} fill="#0f0f1a" stroke="#3f3f46" strokeWidth={1} />
              <text x={150} y={22} textAnchor="middle" fontSize="11" fill="#fbbf24" fontFamily="JetBrains Mono, monospace" fontWeight={600}>
                ready to backprop
              </text>
              <text x={150} y={44} textAnchor="middle" fontSize="10" fill="#a1a1aa" fontFamily="JetBrains Mono, monospace">
                ∂L/∂h_T = h_T − y = {(h[T] - TARGET).toExponential(2)}
              </text>
              <text x={150} y={62} textAnchor="middle" fontSize="10" fill="#6b7280" fontFamily="JetBrains Mono, monospace">
                hit step to walk backward through time
              </text>
            </g>
          )}

          <defs>
            <marker id="arrow-rose" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="#f87171" />
            </marker>
            <marker id="arrow-gray" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="#2a2a32" />
            </marker>
          </defs>
        </svg>
      </div>
    </WidgetFrame>
  )
}
