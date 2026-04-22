'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { SkipBack, StepForward, FastForward } from 'lucide-react'
import { cn } from '@/lib/utils'

// Walk through GPTQ's column-by-column quantization. We:
//   1. build a d×d weight matrix + an inverse-Hessian H⁻¹ diag (sensitivities)
//   2. step through columns in order of decreasing 1/H_ii (highest sensitivity first)
//   3. at each step: quantize that column to int4, compute the residual error,
//      then redistribute a portion of that error to the remaining (un-quantized)
//      columns proportional to their weights in that row — the GPTQ update:
//        W[:, j:] += (error) · H⁻¹[j, j:] / H⁻¹[j, j]
//   4. report running layer-wise MSE
//
// The matrix heatmap recolours as columns flip from fp (amber) → int4 (green),
// and the highlight ring moves to the next column to be processed.

const D = 12 // d×d matrix

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

// int4 quantization with a per-column scale.
function quantCol(col: number[]): number[] {
  const m = Math.max(...col.map(Math.abs))
  if (m === 0) return col.slice()
  const scale = m / 7 // int4 signed range [-8, 7]
  return col.map((v) => Math.max(-8, Math.min(7, Math.round(v / scale))) * scale)
}

type Step = {
  W: number[][]       // current (possibly partially quantized) matrix
  quantized: boolean[] // per column
  orderAll: number[]   // the order columns will be picked in
  justProcessed: number | null
  mse: number
  columnErrors: number[]
}

function buildSteps(): Step[] {
  const rng = mulberry32(11)
  const W0: number[][] = []
  for (let i = 0; i < D; i++) {
    const row: number[] = []
    for (let j = 0; j < D; j++) row.push(gauss(rng) * 0.4)
    W0.push(row)
  }

  // Fake inverse-Hessian diagonals — higher value = more sensitive column
  const hInv: number[] = []
  for (let j = 0; j < D; j++) hInv.push(Math.abs(gauss(rng)) * 0.5 + 0.2)

  // Pick the order: descending sensitivity
  const orderAll = Array.from({ length: D }, (_, i) => i)
    .sort((a, b) => hInv[b] - hInv[a])

  const steps: Step[] = []

  // Step 0: nothing processed
  const initial: Step = {
    W: W0.map((r) => r.slice()),
    quantized: new Array(D).fill(false),
    orderAll,
    justProcessed: null,
    mse: 0,
    columnErrors: new Array(D).fill(0),
  }
  steps.push(initial)

  let W = W0.map((r) => r.slice())
  const quantized = new Array(D).fill(false)
  let runningMse = 0
  const colErrors = new Array(D).fill(0)

  for (let k = 0; k < orderAll.length; k++) {
    const j = orderAll[k]
    // Extract column j, quantize, compute residual
    const colBefore: number[] = W.map((r) => r[j])
    const colAfter = quantCol(colBefore)
    const residual = colBefore.map((v, i) => v - colAfter[i])
    // Write back quantized column
    for (let i = 0; i < D; i++) W[i][j] = colAfter[i]

    // Redistribute error to remaining unquantized columns: for each row i,
    // W[i, j'] += residual[i] * (hInv[j'] / hInv[j]) * alpha  (alpha tunes spread)
    const unprocessed = orderAll.slice(k + 1)
    for (let i = 0; i < D; i++) {
      for (const jp of unprocessed) {
        W[i][jp] += (residual[i] * hInv[jp]) / hInv[j] * 0.25
      }
    }

    quantized[j] = true

    // MSE so far (each quantized column contributes a squared error)
    let colMse = 0
    for (let i = 0; i < D; i++) colMse += residual[i] * residual[i]
    colMse /= D
    colErrors[j] = colMse
    runningMse += colMse

    steps.push({
      W: W.map((r) => r.slice()),
      quantized: quantized.slice(),
      orderAll,
      justProcessed: j,
      mse: runningMse / (k + 1),
      columnErrors: colErrors.slice(),
    })
  }

  return steps
}

const STEPS = buildSteps()
const HINV_ORDER = STEPS[0].orderAll

export default function GPTQDemo() {
  const [i, setI] = useState(0)
  const step = STEPS[i]

  const nextCol = useMemo(() => {
    // The next column that will be quantized on a step-forward
    if (i >= STEPS.length - 1) return null
    const order = step.orderAll
    const done = step.quantized
    for (const idx of order) if (!done[idx]) return idx
    return null
  }, [i, step])

  const absMax = useMemo(() => {
    let m = 0
    for (const row of step.W) for (const v of row) m = Math.max(m, Math.abs(v))
    return Math.max(m, 0.01)
  }, [step])

  const runFull = () => setI(STEPS.length - 1)

  return (
    <WidgetFrame
      widgetName="GPTQDemo"
      label="GPTQ column-by-column — Hessian-ordered, error-compensated"
      right={<span className="font-mono">d = {D} · int4 · W[:, j:] += err · H⁻¹[j, j:]/H⁻¹[j, j]</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <Button onClick={() => setI(0)}>
              <span className="inline-flex items-center gap-1">
                <SkipBack size={11} /> reset
              </span>
            </Button>
            <Button
              onClick={() => setI(Math.min(STEPS.length - 1, i + 1))}
              variant="primary"
              disabled={i === STEPS.length - 1}
            >
              <span className="inline-flex items-center gap-1">
                step <StepForward size={11} />
              </span>
            </Button>
            <Button onClick={runFull} disabled={i === STEPS.length - 1}>
              <span className="inline-flex items-center gap-1">
                run all <FastForward size={11} />
              </span>
            </Button>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="step" value={`${i} / ${STEPS.length - 1}`} />
            <Readout label="running mse" value={step.mse.toExponential(2)} accent="text-term-rose" />
            <Readout
              label="columns done"
              value={`${step.quantized.filter(Boolean).length} / ${D}`}
              accent="text-term-green"
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <div className="grid h-full grid-cols-1 md:grid-cols-[1fr_260px] gap-4 min-h-0">
          {/* Left: the matrix heatmap */}
          <div className="flex flex-col gap-2 min-h-0">
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
              W — amber = fp, green = int4 · ring = next column
            </div>
            <div className="flex-1 min-h-0 flex items-center justify-center">
              <svg viewBox="0 0 520 360" className="w-full h-full">
                <MatrixHeatmap step={step} absMax={absMax} nextCol={nextCol} />
              </svg>
            </div>
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
              1 / H⁻¹[j, j] — higher = more sensitive, quantized first
            </div>
            <HInvBar step={step} nextCol={nextCol} />
          </div>

          {/* Right: processing order + column MSE */}
          <div className="flex flex-col gap-3 min-w-0">
            <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
              column order (descending sensitivity)
            </div>
            <div className="flex-1 min-h-0 overflow-auto rounded border border-dark-border bg-dark-bg/60 p-2">
              <div className="flex flex-col gap-1 font-mono text-[10.5px]">
                {HINV_ORDER.map((j, k) => {
                  const done = step.quantized[j]
                  const isNext = nextCol === j
                  return (
                    <div
                      key={j}
                      className={cn(
                        'flex items-center gap-2 px-2 py-1 rounded',
                        isNext && 'bg-term-amber/10 border border-term-amber',
                        !isNext && done && 'text-term-green',
                        !isNext && !done && 'text-dark-text-muted',
                      )}
                    >
                      <span className="w-5 tabular-nums text-dark-text-disabled">#{k + 1}</span>
                      <span className="w-8 tabular-nums">col {j}</span>
                      <span className="flex-1 tabular-nums text-right">
                        err = {step.columnErrors[j].toExponential(2)}
                      </span>
                      {done && <span className="text-term-green">done</span>}
                      {isNext && <span className="text-term-amber">next</span>}
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function MatrixHeatmap({
  step,
  absMax,
  nextCol,
}: {
  step: Step
  absMax: number
  nextCol: number | null
}) {
  const cellW = 32
  const cellH = 22
  const padL = 40
  const padT = 30

  return (
    <g>
      {/* Column index */}
      {Array.from({ length: D }).map((_, j) => (
        <text
          key={`cj-${j}`}
          x={padL + j * cellW + cellW / 2}
          y={padT - 10}
          textAnchor="middle"
          fontSize="9"
          fill={step.quantized[j] ? '#4ade80' : nextCol === j ? '#fbbf24' : '#666'}
          fontFamily="JetBrains Mono, monospace"
        >
          {j}
        </text>
      ))}

      {/* Row index */}
      {Array.from({ length: D }).map((_, i) => (
        <text
          key={`ri-${i}`}
          x={padL - 8}
          y={padT + i * cellH + cellH / 2 + 3}
          textAnchor="end"
          fontSize="9"
          fill="#666"
          fontFamily="JetBrains Mono, monospace"
        >
          {i}
        </text>
      ))}

      {/* Matrix cells */}
      {step.W.flatMap((row, i) =>
        row.map((v, j) => {
          const mag = Math.min(1, Math.abs(v) / absMax)
          const done = step.quantized[j]
          const color = done ? '74, 222, 128' : '251, 191, 36'
          return (
            <g key={`c-${i}-${j}`}>
              <rect
                x={padL + j * cellW}
                y={padT + i * cellH}
                width={cellW - 1}
                height={cellH - 1}
                fill={`rgba(${color}, ${0.08 + mag * 0.7})`}
                stroke={done ? '#4ade80' : nextCol === j ? '#fbbf24' : '#2a2a32'}
                strokeWidth={nextCol === j ? 1.2 : 0.5}
              />
            </g>
          )
        }),
      )}

      {/* Next-column ring */}
      {nextCol !== null && (
        <rect
          x={padL + nextCol * cellW - 2}
          y={padT - 2}
          width={cellW + 2}
          height={D * cellH + 2}
          fill="none"
          stroke="#fbbf24"
          strokeWidth={1.5}
          rx={3}
        />
      )}

      {/* Just-processed column indicator */}
      {step.justProcessed !== null && (
        <rect
          x={padL + step.justProcessed * cellW - 2}
          y={padT - 2}
          width={cellW + 2}
          height={D * cellH + 2}
          fill="none"
          stroke="#4ade80"
          strokeWidth={1.2}
          strokeDasharray="3 3"
          rx={3}
        />
      )}
    </g>
  )
}

function HInvBar({ step, nextCol }: { step: Step; nextCol: number | null }) {
  return (
    <div
      className="grid gap-[2px] h-8"
      style={{ gridTemplateColumns: `repeat(${D}, 1fr)` }}
    >
      {Array.from({ length: D }).map((_, j) => {
        const orderRank = step.orderAll.indexOf(j)
        // Ranks closer to 0 = higher sensitivity → taller bar
        const norm = 1 - orderRank / (D - 1)
        const done = step.quantized[j]
        const isNext = nextCol === j
        return (
          <div
            key={j}
            className="relative flex items-end justify-center"
            title={`col ${j} — rank ${orderRank + 1}`}
          >
            <div
              className={cn(
                'w-full rounded-sm',
                isNext ? 'bg-term-amber' : done ? 'bg-term-green/60' : 'bg-term-slate/40',
              )}
              style={{ height: `${10 + norm * 70}%` }}
            />
          </div>
        )
      })}
    </div>
  )
}
