'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Pick an operation. See the input shapes, the output shape, the result
// tensor, and the PyTorch code that produced it. Helps readers build the
// reflex of "read the shapes first" before worrying about values.

type Op = 'add' | 'mul' | 'matmul' | 'reshape' | 'sum-dim' | 'broadcast'

interface OpSpec {
  name: Op
  label: string
  a: number[][]
  b?: number[][]
  arg?: string
  code: string
  note: string
  compute: (a: number[][], b?: number[][]) => { data: number[][]; shape: [number, number] }
}

const ones = (r: number, c: number, fill: number = 1) =>
  Array.from({ length: r }, () => Array.from({ length: c }, () => fill))

function dot(a: number[][], b: number[][]): number[][] {
  const out: number[][] = []
  for (let i = 0; i < a.length; i++) {
    const row: number[] = []
    for (let j = 0; j < b[0].length; j++) {
      let s = 0
      for (let k = 0; k < b.length; k++) s += a[i][k] * b[k][j]
      row.push(s)
    }
    out.push(row)
  }
  return out
}

const ops: Record<Op, OpSpec> = {
  add: {
    name: 'add',
    label: 'elementwise add',
    a: [
      [1, 2, 3],
      [4, 5, 6],
    ],
    b: [
      [10, 20, 30],
      [40, 50, 60],
    ],
    code: 'c = a + b',
    note: 'Same-shape tensors add elementwise — C[i,j] = A[i,j] + B[i,j].',
    compute: (a, b) => {
      const data = a.map((row, i) => row.map((v, j) => v + b![i][j]))
      return { data, shape: [a.length, a[0].length] }
    },
  },
  mul: {
    name: 'mul',
    label: 'elementwise multiply (Hadamard)',
    a: [
      [1, 2, 3],
      [4, 5, 6],
    ],
    b: [
      [1, 0, 1],
      [0, 2, 0],
    ],
    code: 'c = a * b',
    note: 'Use the star for elementwise — never for matrix multiplication.',
    compute: (a, b) => {
      const data = a.map((row, i) => row.map((v, j) => v * b![i][j]))
      return { data, shape: [a.length, a[0].length] }
    },
  },
  matmul: {
    name: 'matmul',
    label: 'matrix multiply',
    a: [
      [1, 2],
      [3, 4],
      [5, 6],
    ],
    b: [
      [10, 20, 30],
      [40, 50, 60],
    ],
    code: 'c = a @ b',
    note: '(3 × 2) @ (2 × 3) → (3 × 3).  Inner dims must match; outer dims survive.',
    compute: (a, b) => {
      const data = dot(a, b!)
      return { data, shape: [data.length, data[0].length] }
    },
  },
  reshape: {
    name: 'reshape',
    label: 'reshape',
    a: [
      [1, 2, 3, 4, 5, 6],
    ],
    arg: '(2, 3)',
    code: 'c = a.reshape(2, 3)',
    note: 'Preserves total element count. Values flow in row-major order.',
    compute: (a) => {
      const flat = a.flat()
      const data = [flat.slice(0, 3), flat.slice(3, 6)]
      return { data, shape: [2, 3] }
    },
  },
  'sum-dim': {
    name: 'sum-dim',
    label: 'reduce along a dim',
    a: [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ],
    arg: 'dim=0',
    code: 'c = a.sum(dim=0)',
    note: 'dim=0 sums over rows — collapses the 0-axis, keeps the other. Output shape (3,).',
    compute: (a) => {
      const out: number[] = new Array(a[0].length).fill(0)
      for (const row of a) for (let j = 0; j < row.length; j++) out[j] += row[j]
      return { data: [out], shape: [1, a[0].length] }
    },
  },
  broadcast: {
    name: 'broadcast',
    label: 'broadcasting',
    a: [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ],
    b: [[10, 100, 1000]],
    code: 'c = a + b        # b has shape (1, 3) — expands to (3, 3)',
    note: 'Shapes align from the right. Size 1 dims stretch to match the other.',
    compute: (a, b) => {
      const data = a.map((row) => row.map((v, j) => v + b![0][j]))
      return { data, shape: [a.length, a[0].length] }
    },
  },
}

const OP_KEYS: Op[] = ['add', 'mul', 'matmul', 'reshape', 'sum-dim', 'broadcast']

export default function TensorPlayground() {
  const [op, setOp] = useState<Op>('matmul')
  const spec = ops[op]

  const result = useMemo(() => spec.compute(spec.a, spec.b), [spec])

  return (
    <WidgetFrame
      widgetName="TensorPlayground"
      label="tensor operations — read the shapes first"
      right={<span className="font-mono">all ops elementwise unless marked</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1 flex-wrap">
            {OP_KEYS.map((k) => (
              <button
                key={k}
                onClick={() => setOp(k)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono uppercase transition-all',
                  op === k
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
                )}
              >
                {k}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="out shape"
              value={`(${result.shape[0]}, ${result.shape[1]})`}
              accent="text-term-amber"
            />
            <Readout label="op" value={spec.label} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 overflow-auto">
        <div className="max-w-[880px] mx-auto flex flex-col gap-4 font-mono text-[11.5px]">
          {/* Code line */}
          <div className="rounded border border-dark-border bg-dark-bg px-4 py-2">
            <pre className="text-dark-text-primary">{spec.code}</pre>
          </div>

          {/* Grids */}
          <div className="flex items-center justify-center gap-6 flex-wrap">
            <TensorGrid
              name="a"
              data={spec.a}
              shape={[spec.a.length, spec.a[0].length]}
              color="#67e8f9"
            />
            {spec.b && (
              <>
                <Operator label={opSign(op)} />
                <TensorGrid
                  name="b"
                  data={spec.b}
                  shape={[spec.b.length, spec.b[0].length]}
                  color="#a78bfa"
                />
              </>
            )}
            {spec.arg && (
              <>
                <Operator label="→" />
                <div className="text-dark-text-muted text-[11px] font-mono">{spec.arg}</div>
              </>
            )}
            <Operator label="=" />
            <TensorGrid
              name="c"
              data={result.data}
              shape={result.shape}
              color="#fbbf24"
              isResult
            />
          </div>

          {/* Note */}
          <div className="rounded border border-dark-border bg-dark-surface-elevated/30 p-3 text-[11.5px] font-sans text-dark-text-secondary">
            {spec.note}
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function opSign(op: Op): string {
  if (op === 'add') return '+'
  if (op === 'mul') return '*'
  if (op === 'matmul') return '@'
  if (op === 'broadcast') return '+'
  return '→'
}

function TensorGrid({
  name,
  data,
  shape,
  color,
  isResult,
}: {
  name: string
  data: number[][]
  shape: [number, number]
  color: string
  isResult?: boolean
}) {
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-[12px]" style={{ color }}>
          {name}
        </span>
        <span className="text-[10px] font-mono text-dark-text-disabled">
          ({shape[0]}, {shape[1]})
        </span>
      </div>
      <div
        className={cn('rounded border overflow-hidden', isResult ? 'border-term-amber/60' : 'border-dark-border')}
      >
        <table className="border-collapse">
          <tbody>
            {data.map((row, i) => (
              <tr key={i}>
                {row.map((v, j) => (
                  <td
                    key={j}
                    className={cn(
                      'px-2 py-1 text-center border border-dark-border tabular-nums',
                      isResult ? 'text-term-amber' : 'text-dark-text-primary'
                    )}
                  >
                    {v}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function Operator({ label }: { label: string }) {
  return <span className="text-dark-accent text-lg">{label}</span>
}
