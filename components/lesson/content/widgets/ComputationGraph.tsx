'use client'

import { useState } from 'react'
import WidgetFrame, { Button } from './WidgetFrame'
import { Play, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'

// Visual DAG for L = (a·b + c)·d. Two phases: forward (left → right) fills in
// node values; backward (right → left) fills in gradients. The user steps
// through each phase so the direction of information flow is unmistakable.

interface GraphValues {
  a: number
  b: number
  c: number
  d: number
  ab: number
  e: number
  L: number
  // gradients
  dL: number
  de: number
  dab: number
  dc: number
  dd: number
  da: number
  db: number
}

const initA = 2
const initB = 3
const initC = -1
const initD = 4

function compute(): GraphValues {
  const a = initA
  const b = initB
  const c = initC
  const d = initD
  const ab = a * b
  const e = ab + c
  const L = e * d
  return {
    a,
    b,
    c,
    d,
    ab,
    e,
    L,
    dL: 1,
    de: d,
    dab: d,
    dc: d,
    dd: e,
    da: b * d,
    db: a * d,
  }
}

// Forward steps fill nodes left-to-right. Backward steps fill gradients right-to-left.
// Step 0: just the leaves (a, b, c, d). 1: ab. 2: e. 3: L. 4: dL. 5: de, dd. 6: dab, dc.
// 7: da, db.
const STEPS = 8

export default function ComputationGraph() {
  const [step, setStep] = useState(0)
  const v = compute()

  const nodes = {
    a: step >= 0,
    b: step >= 0,
    c: step >= 0,
    d: step >= 0,
    ab: step >= 1,
    e: step >= 2,
    L: step >= 3,
  }
  const grads = {
    dL: step >= 4,
    de: step >= 5,
    dd: step >= 5,
    dab: step >= 6,
    dc: step >= 6,
    da: step >= 7,
    db: step >= 7,
  }

  const phase =
    step < 4 ? 'forward pass' : step < STEPS ? 'backward pass' : 'complete'
  const phaseColor = step < 4 ? 'text-term-cyan' : 'text-term-rose'

  return (
    <WidgetFrame
      widgetName="ComputationGraph"
      label="computation graph — forward values + backward gradients"
      right={
        <>
          <span className="font-mono">L = (a · b + c) · d</span>
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
          <div className="flex items-center gap-4 ml-auto">
            <span className="text-[11px] font-mono text-dark-text-disabled">
              {step < 4
                ? 'fill values left → right'
                : 'propagate gradients right → left'}
            </span>
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4">
        <svg viewBox="0 0 880 360" className="w-full h-full">
          {/* Edges */}
          <Edge from={[60, 70]} to={[290, 140]} active={nodes.a} color="#67e8f9" />
          <Edge from={[60, 140]} to={[290, 140]} active={nodes.b} color="#67e8f9" />
          <Edge from={[60, 210]} to={[510, 210]} active={nodes.c} color="#67e8f9" />
          <Edge from={[60, 280]} to={[740, 210]} active={nodes.d} color="#67e8f9" />
          <Edge from={[330, 140]} to={[510, 180]} active={nodes.ab} color="#67e8f9" />
          <Edge from={[540, 210]} to={[720, 210]} active={nodes.e} color="#67e8f9" />

          {/* Reverse-edge gradient flow annotations */}
          <BackLabel at={[200, 100]} label={grads.da ? `dL/da=${v.da}` : ''} />
          <BackLabel at={[200, 170]} label={grads.db ? `dL/db=${v.db}` : ''} />
          <BackLabel at={[190, 240]} label={grads.dc ? `dL/dc=${v.dc}` : ''} />
          <BackLabel at={[520, 310]} label={grads.dd ? `dL/dd=${v.dd}` : ''} />
          <BackLabel at={[420, 140]} label={grads.dab ? `dL/d(ab)=${v.dab}` : ''} />
          <BackLabel at={[640, 170]} label={grads.de ? `dL/de=${v.de}` : ''} />
          <BackLabel at={[830, 195]} label={grads.dL ? `dL/dL=${v.dL}` : ''} />

          {/* Nodes */}
          <GraphNode x={60} y={70} label="a" value={nodes.a ? v.a : null} color="#67e8f9" />
          <GraphNode x={60} y={140} label="b" value={nodes.b ? v.b : null} color="#67e8f9" />
          <GraphNode x={60} y={210} label="c" value={nodes.c ? v.c : null} color="#67e8f9" />
          <GraphNode x={60} y={280} label="d" value={nodes.d ? v.d : null} color="#67e8f9" />

          <OpNode x={310} y={140} label="×" valueLabel="a·b" value={nodes.ab ? v.ab : null} />
          <OpNode x={530} y={210} label="+" valueLabel="ab+c" value={nodes.e ? v.e : null} />
          <OpNode x={740} y={210} label="×" valueLabel="L" value={nodes.L ? v.L : null} final />
        </svg>
      </div>
    </WidgetFrame>
  )
}

function Edge({
  from,
  to,
  active,
  color,
}: {
  from: [number, number]
  to: [number, number]
  active: boolean
  color: string
}) {
  return (
    <line
      x1={from[0]}
      y1={from[1]}
      x2={to[0]}
      y2={to[1]}
      stroke={active ? color : '#2a2a2a'}
      strokeWidth={active ? 2 : 1.5}
      strokeOpacity={active ? 0.7 : 0.5}
    />
  )
}

function GraphNode({
  x,
  y,
  label,
  value,
  color,
}: {
  x: number
  y: number
  label: string
  value: number | null
  color: string
}) {
  return (
    <g>
      <circle cx={x} cy={y} r={22} fill="#1f1b3a" stroke={color} strokeWidth={1.5} />
      <text
        x={x}
        y={y - 2}
        textAnchor="middle"
        fontSize="12"
        fill="#ccc"
        fontFamily="JetBrains Mono, monospace"
        fontWeight="600"
      >
        {label}
      </text>
      <text
        x={x}
        y={y + 12}
        textAnchor="middle"
        fontSize="10"
        fill={value !== null ? '#fbbf24' : '#444'}
        fontFamily="JetBrains Mono, monospace"
      >
        {value !== null ? `=${value}` : '?'}
      </text>
    </g>
  )
}

function OpNode({
  x,
  y,
  label,
  valueLabel,
  value,
  final,
}: {
  x: number
  y: number
  label: string
  valueLabel: string
  value: number | null
  final?: boolean
}) {
  return (
    <g>
      <rect
        x={x - 28}
        y={y - 22}
        width={56}
        height={44}
        rx={6}
        fill="#1a1a1a"
        stroke={final ? '#fbbf24' : '#a78bfa'}
        strokeWidth={1.5}
      />
      <text
        x={x}
        y={y - 6}
        textAnchor="middle"
        fontSize="13"
        fill={final ? '#fbbf24' : '#a78bfa'}
        fontFamily="JetBrains Mono, monospace"
        fontWeight="700"
      >
        {label}
      </text>
      <text
        x={x}
        y={y + 12}
        textAnchor="middle"
        fontSize="10"
        fill={value !== null ? (final ? '#fbbf24' : '#fbbf24') : '#444'}
        fontFamily="JetBrains Mono, monospace"
      >
        {value !== null ? `${valueLabel}=${value}` : valueLabel}
      </text>
    </g>
  )
}

function BackLabel({ at, label }: { at: [number, number]; label: string }) {
  if (!label) return null
  return (
    <g>
      <rect
        x={at[0] - 52}
        y={at[1] - 10}
        width={104}
        height={18}
        rx={4}
        fill="#0a0a0a"
        stroke="#f472b6"
        strokeWidth={1}
        opacity={0.9}
      />
      <text
        x={at[0]}
        y={at[1] + 3}
        textAnchor="middle"
        fontSize="10"
        fill="#f472b6"
        fontFamily="JetBrains Mono, monospace"
      >
        {label}
      </text>
    </g>
  )
}
