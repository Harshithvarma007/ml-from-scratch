'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { Play, Pause, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useEffect } from 'react'

type Mode = 'max' | 'avg'

const IN = 8
const STRIDE = 2
const K = 2
const OUT = Math.floor((IN - K) / STRIDE) + 1

function mulberry32(seed: number) {
  let t = seed >>> 0
  return () => {
    t += 0x6d2b79f5
    let x = Math.imul(t ^ (t >>> 15), t | 1)
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61)
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296
  }
}

function makeInput(): number[][] {
  const rng = mulberry32(13)
  return Array.from({ length: IN }, () => Array.from({ length: IN }, () => Math.floor(rng() * 10)))
}

function pool(input: number[][], mode: Mode): number[][] {
  const out: number[][] = []
  for (let y = 0; y < OUT; y++) {
    const row: number[] = []
    for (let x = 0; x < OUT; x++) {
      const values: number[] = []
      for (let ky = 0; ky < K; ky++) {
        for (let kx = 0; kx < K; kx++) {
          values.push(input[y * STRIDE + ky][x * STRIDE + kx])
        }
      }
      row.push(mode === 'max' ? Math.max(...values) : values.reduce((a, b) => a + b, 0) / values.length)
    }
    out.push(row)
  }
  return out
}

export default function PoolingViz() {
  const [mode, setMode] = useState<Mode>('max')
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)

  const input = useMemo(() => makeInput(), [])
  const output = useMemo(() => pool(input, mode), [input, mode])
  const total = OUT * OUT

  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => setStep((s) => (s + 1) % total), 600)
    return () => clearInterval(id)
  }, [playing, total])

  const cy = Math.floor(step / OUT)
  const cx = step % OUT

  const winVals: number[] = []
  for (let ky = 0; ky < K; ky++) {
    for (let kx = 0; kx < K; kx++) {
      winVals.push(input[cy * STRIDE + ky][cx * STRIDE + kx])
    }
  }
  const winOut = mode === 'max' ? Math.max(...winVals) : winVals.reduce((a, b) => a + b, 0) / winVals.length

  const isWin = (y: number, x: number) => {
    const dy = y - cy * STRIDE, dx = x - cx * STRIDE
    return dy >= 0 && dy < K && dx >= 0 && dx < K
  }
  const isWinMax = (y: number, x: number) => {
    if (!isWin(y, x)) return false
    return input[y][x] === Math.max(...winVals)
  }

  return (
    <WidgetFrame
      widgetName="PoolingViz"
      label="pooling — a 2×2 window strides across, collapsing each block"
      right={<span className="font-mono">in ({IN}×{IN}) · k={K} · stride={STRIDE} → out ({OUT}×{OUT})</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            {(['max', 'avg'] as Mode[]).map((m) => (
              <button key={m} onClick={() => setMode(m)} className={cn('px-2.5 py-1 rounded text-[11px] font-mono uppercase transition-all', mode === m ? 'bg-dark-accent text-white' : 'border border-dark-border text-dark-text-secondary')}>
                {m} pool
              </button>
            ))}
          </div>
          <Button onClick={() => setPlaying((p) => !p)} variant="primary">
            {playing ? <><Pause className="w-3 h-3 inline -mt-px mr-1" /> pause</> : <><Play className="w-3 h-3 inline -mt-px mr-1" /> play</>}
          </Button>
          <Button onClick={() => { setStep(0); setPlaying(false) }}>
            <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
          </Button>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="window" value={`[${winVals.join(', ')}]`} />
            <Readout label="→ out" value={winOut.toFixed(mode === 'avg' ? 2 : 0)} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 flex items-center justify-center gap-10 overflow-auto">
        <div className="flex flex-col items-center gap-2">
          <div className="text-[10px] font-mono text-dark-text-disabled uppercase tracking-wider">input</div>
          <div className="inline-flex flex-col gap-1">
            {input.map((row, y) => (
              <div key={y} className="flex gap-1">
                {row.map((v, x) => {
                  const win = isWin(y, x)
                  const max = isWinMax(y, x)
                  return (
                    <div
                      key={x}
                      className={cn(
                        'w-9 h-9 rounded text-[11px] font-mono flex items-center justify-center tabular-nums transition-all',
                        max && mode === 'max' ? 'bg-term-amber/70 ring-2 ring-term-amber text-dark-bg font-bold' :
                        win ? 'bg-term-amber/25 ring-1 ring-term-amber/60 text-dark-text-primary' :
                        'bg-dark-surface-elevated/40 text-dark-text-muted'
                      )}
                    >
                      {v}
                    </div>
                  )
                })}
              </div>
            ))}
          </div>
        </div>

        <div className="flex flex-col items-center gap-2">
          <div className="text-[10px] font-mono text-dark-text-disabled uppercase tracking-wider">output</div>
          <div className="inline-flex flex-col gap-1">
            {output.map((row, y) => (
              <div key={y} className="flex gap-1">
                {row.map((v, x) => {
                  const cur = y === cy && x === cx
                  return (
                    <div
                      key={x}
                      onClick={() => { setStep(y * OUT + x); setPlaying(false) }}
                      className={cn(
                        'w-9 h-9 rounded text-[11px] font-mono flex items-center justify-center tabular-nums cursor-pointer transition-all',
                        cur ? 'bg-term-amber/60 ring-2 ring-term-amber text-dark-bg font-bold' : 'bg-dark-surface-elevated/40 text-dark-text-muted hover:bg-dark-surface-elevated/60'
                      )}
                    >
                      {mode === 'avg' ? v.toFixed(1) : v.toFixed(0)}
                    </div>
                  )
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
