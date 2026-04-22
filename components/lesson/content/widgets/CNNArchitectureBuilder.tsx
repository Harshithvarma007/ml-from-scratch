'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'
import { Plus, X } from 'lucide-react'

// Build a tiny CNN layer-by-layer. Pick blocks from the palette; they append
// to the stack. Shape arithmetic propagates through; param count + FLOPs
// update live. Two preset buttons: "load LeNet" and "load VGG-style".

type Block =
  | { kind: 'conv'; channels: number; k: number; stride: number; padding: number }
  | { kind: 'pool'; k: number; stride: number }
  | { kind: 'fc'; out: number }

interface Shape {
  C: number
  H: number
  W: number
}

const INPUT: Shape = { C: 3, H: 32, W: 32 }

function forward(blocks: Block[]): (Shape & { params: number; flops: number; kind: string })[] {
  const out: (Shape & { params: number; flops: number; kind: string })[] = [
    { ...INPUT, params: 0, flops: 0, kind: 'input' },
  ]
  for (const b of blocks) {
    const prev = out[out.length - 1]
    if (b.kind === 'conv') {
      const H = Math.floor((prev.H + 2 * b.padding - b.k) / b.stride) + 1
      const W = Math.floor((prev.W + 2 * b.padding - b.k) / b.stride) + 1
      const params = prev.C * b.channels * b.k * b.k + b.channels
      const flops = 2 * H * W * b.channels * prev.C * b.k * b.k
      out.push({ C: b.channels, H: Math.max(0, H), W: Math.max(0, W), params, flops, kind: `conv ${b.k}×${b.k}` })
    } else if (b.kind === 'pool') {
      const H = Math.floor((prev.H - b.k) / b.stride) + 1
      const W = Math.floor((prev.W - b.k) / b.stride) + 1
      out.push({ C: prev.C, H: Math.max(0, H), W: Math.max(0, W), params: 0, flops: 0, kind: `pool ${b.k}×${b.k}` })
    } else {
      const flatIn = prev.C * prev.H * prev.W
      out.push({ C: b.out, H: 1, W: 1, params: flatIn * b.out + b.out, flops: 2 * flatIn * b.out, kind: `fc ${b.out}` })
    }
  }
  return out
}

const LENET: Block[] = [
  { kind: 'conv', channels: 6, k: 5, stride: 1, padding: 0 },
  { kind: 'pool', k: 2, stride: 2 },
  { kind: 'conv', channels: 16, k: 5, stride: 1, padding: 0 },
  { kind: 'pool', k: 2, stride: 2 },
  { kind: 'fc', out: 120 },
  { kind: 'fc', out: 84 },
  { kind: 'fc', out: 10 },
]

const VGGISH: Block[] = [
  { kind: 'conv', channels: 32, k: 3, stride: 1, padding: 1 },
  { kind: 'conv', channels: 32, k: 3, stride: 1, padding: 1 },
  { kind: 'pool', k: 2, stride: 2 },
  { kind: 'conv', channels: 64, k: 3, stride: 1, padding: 1 },
  { kind: 'conv', channels: 64, k: 3, stride: 1, padding: 1 },
  { kind: 'pool', k: 2, stride: 2 },
  { kind: 'fc', out: 128 },
  { kind: 'fc', out: 10 },
]

function fmt(n: number): string {
  if (n < 1000) return String(n)
  if (n < 1e6) return (n / 1e3).toFixed(1) + 'K'
  if (n < 1e9) return (n / 1e6).toFixed(1) + 'M'
  return (n / 1e9).toFixed(2) + 'G'
}

export default function CNNArchitectureBuilder() {
  const [blocks, setBlocks] = useState<Block[]>(LENET)
  const stages = useMemo(() => forward(blocks), [blocks])
  const totalParams = stages.reduce((s, x) => s + x.params, 0)
  const totalFlops = stages.reduce((s, x) => s + x.flops, 0)

  const add = (b: Block) => setBlocks((prev) => [...prev, b])
  const remove = (i: number) => setBlocks((prev) => prev.filter((_, j) => j !== i))

  return (
    <WidgetFrame
      widgetName="CNNArchitectureBuilder"
      label="build a CNN — watch shapes, params, and FLOPs"
      right={
        <>
          <span className="font-mono">input: ({INPUT.C}, {INPUT.H}, {INPUT.W}) = CIFAR-style</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Button onClick={() => setBlocks(LENET)} variant="primary">
            load LeNet
          </Button>
          <Button onClick={() => setBlocks(VGGISH)}>load VGG-lite</Button>
          <Button onClick={() => setBlocks([])}>clear</Button>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="params" value={fmt(totalParams)} accent="text-term-amber" />
            <Readout label="FLOPs / fwd" value={fmt(totalFlops)} accent="text-term-cyan" />
            <Readout label="output" value={stages.length ? `(${stages[stages.length - 1].C}, ${stages[stages.length - 1].H}, ${stages[stages.length - 1].W})` : '—'} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[220px_1fr] gap-4 overflow-auto">
        {/* Palette */}
        <div className="rounded border border-dark-border bg-dark-bg p-3">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
            add a block
          </div>
          <div className="space-y-1.5 text-[11px] font-mono">
            {[
              { label: 'conv 3×3 → 32ch', b: { kind: 'conv', channels: 32, k: 3, stride: 1, padding: 1 } as Block },
              { label: 'conv 3×3 → 64ch', b: { kind: 'conv', channels: 64, k: 3, stride: 1, padding: 1 } as Block },
              { label: 'conv 3×3 → 128ch', b: { kind: 'conv', channels: 128, k: 3, stride: 1, padding: 1 } as Block },
              { label: 'conv 5×5 → 32ch', b: { kind: 'conv', channels: 32, k: 5, stride: 1, padding: 0 } as Block },
              { label: 'pool 2×2', b: { kind: 'pool', k: 2, stride: 2 } as Block },
              { label: 'fc → 128', b: { kind: 'fc', out: 128 } as Block },
              { label: 'fc → 10', b: { kind: 'fc', out: 10 } as Block },
            ].map((p, i) => (
              <button
                key={i}
                onClick={() => add(p.b)}
                className="w-full flex items-center justify-between px-2 py-1.5 rounded border border-dark-border hover:border-dark-accent/60 hover:bg-white/[0.02] text-dark-text-secondary transition-all"
              >
                <span>{p.label}</span>
                <Plus className="w-3 h-3 text-dark-accent" />
              </button>
            ))}
          </div>
        </div>

        {/* Stack */}
        <div className="rounded border border-dark-border bg-dark-bg p-3 overflow-auto">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
            the network ({blocks.length} blocks)
          </div>
          <table className="w-full font-mono text-[11px] tabular-nums">
            <thead className="text-dark-text-disabled text-[10px] uppercase">
              <tr>
                <th className="text-left pb-1">#</th>
                <th className="text-left pb-1">layer</th>
                <th className="text-right pb-1">shape</th>
                <th className="text-right pb-1">params</th>
                <th className="text-right pb-1">FLOPs</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {stages.map((s, i) => (
                <tr key={i} className={cn('border-t border-dark-border/60', s.H === 0 && 'text-term-rose')}>
                  <td className="py-1 text-dark-text-disabled">{i}</td>
                  <td
                    className={cn(
                      'py-1',
                      s.kind === 'input'
                        ? 'text-term-cyan'
                        : s.kind.startsWith('conv')
                          ? 'text-term-purple'
                          : s.kind.startsWith('pool')
                            ? 'text-term-amber'
                            : 'text-term-green',
                    )}
                  >
                    {s.kind}
                  </td>
                  <td className="py-1 text-right text-dark-text-primary">
                    ({s.C}, {s.H}, {s.W})
                  </td>
                  <td className="py-1 text-right text-dark-text-muted">{fmt(s.params)}</td>
                  <td className="py-1 text-right text-dark-text-muted">{fmt(s.flops)}</td>
                  <td className="py-1 text-right">
                    {i > 0 && (
                      <button
                        onClick={() => remove(i - 1)}
                        className="text-dark-text-disabled hover:text-term-rose"
                      >
                        <X className="w-3 h-3 inline" />
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </WidgetFrame>
  )
}
