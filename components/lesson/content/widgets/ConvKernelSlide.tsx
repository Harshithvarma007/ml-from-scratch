'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Button, Readout } from './WidgetFrame'
import { Play, Pause, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'

// Animated 2D conv: a 3x3 kernel slides over a 7x7 input, producing a 5x5
// output. The highlighted cell in the output shows the current window × kernel
// element-wise product sum. Toggle auto-play, adjust stride/padding, watch
// the output shape change.

type Kernel = 'edge-h' | 'edge-v' | 'blur' | 'sharpen'

const KERNELS: Record<Kernel, number[][]> = {
  'edge-h': [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
  'edge-v': [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
  'blur':   [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]],
  'sharpen':[[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
}

const IN_H = 7, IN_W = 7, K = 3

function makeInput(): number[][] {
  // A small grayscale "digit-ish" pattern
  return [
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 1, 1],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
  ]
}

function convolve(input: number[][], kernel: number[][], stride: number, padding: number): number[][] {
  const inH = input.length, inW = input[0].length
  const kH = kernel.length, kW = kernel[0].length
  const outH = Math.floor((inH + 2 * padding - kH) / stride) + 1
  const outW = Math.floor((inW + 2 * padding - kW) / stride) + 1
  const out: number[][] = []
  for (let y = 0; y < outH; y++) {
    const row: number[] = []
    for (let x = 0; x < outW; x++) {
      let s = 0
      for (let ky = 0; ky < kH; ky++) {
        for (let kx = 0; kx < kW; kx++) {
          const iy = y * stride + ky - padding
          const ix = x * stride + kx - padding
          const v = iy >= 0 && iy < inH && ix >= 0 && ix < inW ? input[iy][ix] : 0
          s += v * kernel[ky][kx]
        }
      }
      row.push(s)
    }
    out.push(row)
  }
  return out
}

export default function ConvKernelSlide() {
  const [kernelName, setKernelName] = useState<Kernel>('edge-v')
  const [stride, setStride] = useState(1)
  const [padding, setPadding] = useState(0)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)

  const input = useMemo(() => makeInput(), [])
  const kernel = KERNELS[kernelName]
  const output = useMemo(() => convolve(input, kernel, stride, padding), [input, kernel, stride, padding])

  const outH = output.length
  const outW = output[0].length
  const total = outH * outW

  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => setStep((s) => (s + 1) % (total + 1)), 400)
    return () => clearInterval(id)
  }, [playing, total])

  useEffect(() => {
    setStep(0)
    setPlaying(false)
  }, [kernelName, stride, padding])

  const cursorY = Math.min(step, total - 1) >= 0 ? Math.floor(Math.min(step, total - 1) / outW) : 0
  const cursorX = Math.min(step, total - 1) >= 0 ? Math.min(step, total - 1) % outW : 0

  const currentProduct = () => {
    const terms: { iy: number, ix: number, v: number, kv: number }[] = []
    let s = 0
    for (let ky = 0; ky < K; ky++) {
      for (let kx = 0; kx < K; kx++) {
        const iy = cursorY * stride + ky - padding
        const ix = cursorX * stride + kx - padding
        const v = iy >= 0 && iy < IN_H && ix >= 0 && ix < IN_W ? input[iy][ix] : 0
        terms.push({ iy, ix, v, kv: kernel[ky][kx] })
        s += v * kernel[ky][kx]
      }
    }
    return { terms, sum: s }
  }
  const { terms, sum } = currentProduct()

  const inHighlight = (y: number, x: number) => {
    const ty = y + padding - cursorY * stride
    const tx = x + padding - cursorX * stride
    return ty >= 0 && ty < K && tx >= 0 && tx < K
  }

  return (
    <WidgetFrame
      widgetName="ConvKernelSlide"
      label="convolution — a kernel slides, multiplies, sums"
      right={<span className="font-mono">out = ⌊(in + 2p − k) / s⌋ + 1 · (in={IN_H}, k=3, s={stride}, p={padding}) → ({outH}, {outW})</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            {(Object.keys(KERNELS) as Kernel[]).map((k) => (
              <button
                key={k}
                onClick={() => setKernelName(k)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
                  kernelName === k ? 'bg-dark-accent text-white' : 'border border-dark-border text-dark-text-secondary'
                )}
              >
                {k}
              </button>
            ))}
          </div>
          <Slider label="stride" value={stride} min={1} max={3} step={1} onChange={(v) => setStride(Math.round(v))} format={(v) => String(Math.round(v))} accent="accent-term-purple" />
          <Slider label="padding" value={padding} min={0} max={2} step={1} onChange={(v) => setPadding(Math.round(v))} format={(v) => String(Math.round(v))} accent="accent-term-green" />
          <Button onClick={() => setPlaying((p) => !p)} variant="primary">
            {playing ? <><Pause className="w-3 h-3 inline -mt-px mr-1" /> pause</> : <><Play className="w-3 h-3 inline -mt-px mr-1" /> play</>}
          </Button>
          <Button onClick={() => { setStep(0); setPlaying(false) }}>
            <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
          </Button>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="out@cursor" value={output[cursorY]?.[cursorX]?.toFixed(2) ?? '—'} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 flex items-center gap-6 overflow-auto">
        {/* Input grid */}
        <div className="flex flex-col gap-1">
          <div className="text-[10px] font-mono text-dark-text-disabled uppercase tracking-wider text-center">input ({IN_H}×{IN_W})</div>
          <div className="inline-flex flex-col gap-1">
            {input.map((row, y) => (
              <div key={y} className="flex gap-1">
                {row.map((v, x) => {
                  const hl = inHighlight(y, x)
                  return (
                    <div
                      key={x}
                      className={cn(
                        'w-7 h-7 rounded text-[10px] font-mono flex items-center justify-center transition-all',
                        hl ? 'bg-term-amber/40 ring-1 ring-term-amber' : 'bg-dark-surface-elevated/40'
                      )}
                      style={{ color: v > 0.5 ? '#fff' : '#666' }}
                    >
                      {v.toFixed(0)}
                    </div>
                  )
                })}
              </div>
            ))}
          </div>
        </div>

        {/* Kernel */}
        <div className="flex flex-col gap-1">
          <div className="text-[10px] font-mono text-dark-text-disabled uppercase tracking-wider text-center">kernel</div>
          <div className="inline-flex flex-col gap-1">
            {kernel.map((row, y) => (
              <div key={y} className="flex gap-1">
                {row.map((v, x) => (
                  <div key={x} className="w-7 h-7 rounded text-[9px] font-mono flex items-center justify-center bg-term-purple/30 text-term-purple tabular-nums">
                    {v.toFixed(1)}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* Output */}
        <div className="flex flex-col gap-1">
          <div className="text-[10px] font-mono text-dark-text-disabled uppercase tracking-wider text-center">output ({outH}×{outW})</div>
          <div className="inline-flex flex-col gap-1">
            {output.map((row, y) => (
              <div key={y} className="flex gap-1">
                {row.map((v, x) => {
                  const isCursor = y === cursorY && x === cursorX
                  return (
                    <div
                      key={x}
                      onClick={() => { setStep(y * outW + x); setPlaying(false) }}
                      className={cn(
                        'w-7 h-7 rounded text-[10px] font-mono flex items-center justify-center cursor-pointer transition-all',
                        isCursor ? 'bg-term-amber/60 ring-1 ring-term-amber' : 'bg-dark-surface-elevated/40 hover:bg-dark-surface-elevated/60'
                      )}
                      style={{ color: isCursor ? '#fff' : '#aaa' }}
                    >
                      {v.toFixed(1)}
                    </div>
                  )
                })}
              </div>
            ))}
          </div>
        </div>

        {/* Current computation */}
        <div className="ml-auto w-[240px] rounded border border-dark-border bg-dark-surface-elevated/40 p-3 font-mono text-[10.5px]">
          <div className="text-dark-text-disabled uppercase tracking-wider text-[10px] mb-2">output[{cursorY}][{cursorX}] =</div>
          <div className="space-y-0.5 text-dark-text-muted">
            {terms.slice(0, 4).map((t, i) => (
              <div key={i}>{t.v.toFixed(1)} · {t.kv.toFixed(1)}{i < terms.length - 1 ? ' +' : ''}</div>
            ))}
            <div className="text-dark-text-disabled">... + 5 more</div>
          </div>
          <div className="mt-2 pt-2 border-t border-dark-border text-term-amber font-semibold">= {sum.toFixed(2)}</div>
        </div>
      </div>
    </WidgetFrame>
  )
}
