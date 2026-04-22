'use client'

import { useState } from 'react'
import WidgetFrame, { Button } from './WidgetFrame'
import { Play, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'

// Pick a scalar PyTorch expression. See the computation graph PyTorch
// silently builds. Step through forward values filling in, then the backward
// pass walking the graph in reverse. Locks in "autograd is a graph."

interface NodeSpec {
  id: string
  label: string
  op?: string
  value: number
  grad?: number // dL/dnode after backward
  x: number
  y: number
}

interface EdgeSpec {
  from: string
  to: string
}

interface Expr {
  name: string
  pyCode: string
  nodes: NodeSpec[]
  edges: EdgeSpec[]
  // Which step index reveals which node (for forward) and which grad (for backward).
  forwardOrder: string[] // node ids
  backwardOrder: string[] // node ids
}

// Three example expressions.
const exprs: Expr[] = [
  // 1) L = (x + y) * z,  x=2, y=3, z=4
  {
    name: '(x + y) · z',
    pyCode: `x = torch.tensor(2., requires_grad=True)
y = torch.tensor(3., requires_grad=True)
z = torch.tensor(4., requires_grad=True)

L = (x + y) * z
L.backward()`,
    nodes: [
      { id: 'x', label: 'x', value: 2, grad: 4, x: 90, y: 90 },
      { id: 'y', label: 'y', value: 3, grad: 4, x: 90, y: 170 },
      { id: 'z', label: 'z', value: 4, grad: 5, x: 90, y: 250 },
      { id: 's', label: '+', op: 'Add', value: 5, grad: 4, x: 320, y: 130 },
      { id: 'L', label: '×', op: 'Mul', value: 20, grad: 1, x: 560, y: 170 },
    ],
    edges: [
      { from: 'x', to: 's' },
      { from: 'y', to: 's' },
      { from: 's', to: 'L' },
      { from: 'z', to: 'L' },
    ],
    forwardOrder: ['x', 'y', 'z', 's', 'L'],
    backwardOrder: ['L', 's', 'z', 'x', 'y'],
  },
  // 2) L = sum(x^2),  x = [1, 2, 3]
  {
    name: 'sum(x²) — a simple loss',
    pyCode: `x = torch.tensor([1., 2., 3.], requires_grad=True)

L = (x ** 2).sum()
L.backward()`,
    nodes: [
      { id: 'x0', label: 'x₀', value: 1, grad: 2, x: 90, y: 80 },
      { id: 'x1', label: 'x₁', value: 2, grad: 4, x: 90, y: 170 },
      { id: 'x2', label: 'x₂', value: 3, grad: 6, x: 90, y: 260 },
      { id: 's0', label: '()²', op: 'Pow', value: 1, grad: 1, x: 300, y: 80 },
      { id: 's1', label: '()²', op: 'Pow', value: 4, grad: 1, x: 300, y: 170 },
      { id: 's2', label: '()²', op: 'Pow', value: 9, grad: 1, x: 300, y: 260 },
      { id: 'L', label: 'Σ', op: 'Sum', value: 14, grad: 1, x: 540, y: 170 },
    ],
    edges: [
      { from: 'x0', to: 's0' },
      { from: 'x1', to: 's1' },
      { from: 'x2', to: 's2' },
      { from: 's0', to: 'L' },
      { from: 's1', to: 'L' },
      { from: 's2', to: 'L' },
    ],
    forwardOrder: ['x0', 'x1', 'x2', 's0', 's1', 's2', 'L'],
    backwardOrder: ['L', 's0', 's1', 's2', 'x0', 'x1', 'x2'],
  },
  // 3) L = (w * x + b - y)^2
  {
    name: '(wx + b − y)² — a regression step',
    pyCode: `w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
x, y = torch.tensor(2.), torch.tensor(2.)  # no grad: data

L = (w * x + b - y) ** 2
L.backward()  # grads only on w and b`,
    nodes: [
      { id: 'w', label: 'w', value: 0.5, grad: -3.6, x: 70, y: 80 },
      { id: 'x', label: 'x', value: 2, x: 70, y: 165 },
      { id: 'b', label: 'b', value: 0.1, grad: -1.8, x: 70, y: 250 },
      { id: 'y', label: 'y', value: 2, x: 70, y: 320 },
      { id: 'wx', label: '×', op: 'Mul', value: 1, grad: -1.8, x: 280, y: 130 },
      { id: 'sum', label: '+', op: 'Add', value: 1.1, grad: -1.8, x: 430, y: 200 },
      { id: 'diff', label: '−', op: 'Sub', value: -0.9, grad: -1.8, x: 560, y: 200 },
      { id: 'L', label: '()²', op: 'Pow', value: 0.81, grad: 1, x: 690, y: 200 },
    ],
    edges: [
      { from: 'w', to: 'wx' },
      { from: 'x', to: 'wx' },
      { from: 'wx', to: 'sum' },
      { from: 'b', to: 'sum' },
      { from: 'sum', to: 'diff' },
      { from: 'y', to: 'diff' },
      { from: 'diff', to: 'L' },
    ],
    forwardOrder: ['w', 'x', 'b', 'y', 'wx', 'sum', 'diff', 'L'],
    backwardOrder: ['L', 'diff', 'sum', 'wx', 'w', 'b'],
  },
]

export default function AutogradTrace() {
  const [idx, setIdx] = useState(0)
  const [step, setStep] = useState(0)

  const expr = exprs[idx]
  const TOTAL = expr.forwardOrder.length + expr.backwardOrder.length

  const revealedForward = new Set(expr.forwardOrder.slice(0, Math.min(step, expr.forwardOrder.length)))
  const revealedBackward = new Set(
    step > expr.forwardOrder.length
      ? expr.backwardOrder.slice(0, step - expr.forwardOrder.length)
      : [],
  )

  const phase =
    step < expr.forwardOrder.length
      ? 'forward pass'
      : step < TOTAL
        ? 'backward pass'
        : 'complete'
  const phaseColor =
    step < expr.forwardOrder.length ? 'text-term-cyan' : 'text-term-rose'

  const pickExpr = (i: number) => {
    setIdx(i)
    setStep(0)
  }

  return (
    <WidgetFrame
      widgetName="AutogradTrace"
      label="autograd — the computation graph PyTorch silently builds"
      right={
        <>
          <span className="font-mono">expression: {expr.name}</span>
          <span className="text-dark-text-disabled">·</span>
          <span className={cn('font-mono', phaseColor)}>{phase}</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            {exprs.map((e, i) => (
              <button
                key={i}
                onClick={() => pickExpr(i)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono transition-all max-w-[160px] truncate',
                  idx === i
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
                )}
              >
                {e.name}
              </button>
            ))}
          </div>
          <Button onClick={() => setStep((s) => Math.min(TOTAL, s + 1))} variant="primary" disabled={step >= TOTAL}>
            <Play className="w-3 h-3 inline -mt-px mr-1" /> next
          </Button>
          <Button onClick={() => setStep(0)}>
            <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
          </Button>
          <span className="text-[11px] font-mono text-dark-text-muted">
            step {step} / {TOTAL}
          </span>
        </div>
      }
    >
      <div className="absolute inset-0 grid grid-cols-1 md:grid-cols-[1fr_320px] gap-2 p-2">
        {/* Graph */}
        <div className="border border-dark-border rounded bg-dark-bg relative overflow-hidden">
          <svg viewBox="0 0 800 400" className="w-full h-full">
            {/* Edges */}
            {expr.edges.map((e, i) => {
              const a = expr.nodes.find((n) => n.id === e.from)!
              const b = expr.nodes.find((n) => n.id === e.to)!
              const fwdReached = revealedForward.has(e.to)
              const backReached = revealedBackward.has(e.from)
              const color = backReached ? '#f472b6' : fwdReached ? '#67e8f9' : '#2a2a2a'
              const opacity = fwdReached || backReached ? 0.75 : 0.3
              return (
                <line
                  key={i}
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  stroke={color}
                  strokeOpacity={opacity}
                  strokeWidth={1.5}
                />
              )
            })}

            {/* Nodes */}
            {expr.nodes.map((n) => {
              const fw = revealedForward.has(n.id)
              const bk = revealedBackward.has(n.id)
              const isOp = Boolean(n.op)
              return (
                <g key={n.id}>
                  {isOp ? (
                    <rect
                      x={n.x - 22}
                      y={n.y - 18}
                      width={44}
                      height={36}
                      rx={5}
                      fill="#1a1a1a"
                      stroke={fw ? '#a78bfa' : '#444'}
                      strokeWidth={1.5}
                    />
                  ) : (
                    <circle
                      cx={n.x}
                      cy={n.y}
                      r={20}
                      fill="#1f1b3a"
                      stroke={fw ? '#67e8f9' : '#444'}
                      strokeWidth={1.5}
                    />
                  )}
                  <text
                    x={n.x}
                    y={n.y - 2}
                    textAnchor="middle"
                    fontSize="11"
                    fill={fw ? '#ccc' : '#555'}
                    fontFamily="JetBrains Mono, monospace"
                    fontWeight="600"
                  >
                    {n.label}
                  </text>
                  {fw && (
                    <text
                      x={n.x}
                      y={n.y + 11}
                      textAnchor="middle"
                      fontSize="9"
                      fill="#fbbf24"
                      fontFamily="JetBrains Mono, monospace"
                    >
                      {formatValue(n.value)}
                    </text>
                  )}
                  {bk && n.grad !== undefined && (
                    <g>
                      <rect
                        x={n.x + 26}
                        y={n.y - 10}
                        width={58}
                        height={18}
                        rx={4}
                        fill="#0a0a0a"
                        stroke="#f472b6"
                        strokeWidth={1}
                        opacity={0.92}
                      />
                      <text
                        x={n.x + 55}
                        y={n.y + 3}
                        textAnchor="middle"
                        fontSize="9"
                        fill="#f472b6"
                        fontFamily="JetBrains Mono, monospace"
                      >
                        ∂L={formatValue(n.grad)}
                      </text>
                    </g>
                  )}
                </g>
              )
            })}
          </svg>
        </div>

        {/* Code + legend */}
        <div className="flex flex-col gap-2 min-h-0">
          <div className="border border-dark-border rounded bg-dark-bg p-3 flex-1 overflow-auto">
            <div className="text-[10px] uppercase tracking-wider text-dark-text-disabled mb-1">
              pytorch code
            </div>
            <pre className="font-mono text-[11.5px] text-dark-text-primary whitespace-pre leading-relaxed">
              {expr.pyCode}
            </pre>
          </div>
          <div className="border border-dark-border rounded bg-dark-bg p-3 text-[10.5px] font-mono text-dark-text-muted space-y-1">
            <div className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-full bg-term-cyan/60" />
              <span>leaf tensor (data or parameter)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded bg-term-purple/60" />
              <span>op node — stores forward math + backward rule</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded bg-term-pink/60" />
              <span>δ bubble = .grad after loss.backward()</span>
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function formatValue(v: number): string {
  if (Math.abs(v) >= 100) return v.toFixed(0)
  if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1)
  return v.toString()
}
