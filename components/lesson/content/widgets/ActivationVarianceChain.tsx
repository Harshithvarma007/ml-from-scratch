'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// For three init strategies, simulate a chain of L layers and plot the
// variance of activations at each depth. Naive blows up or collapses. Xavier
// works for tanh. He works for ReLU. The three curves on one plot make this
// vivid.

type Act = 'tanh' | 'relu'
type Init = 'naive' | 'xavier' | 'he'

const WIDTH = 128
const N_SAMPLES = 256
const MAX_L = 30

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

function variance(arr: number[]): number {
  const m = arr.reduce((s, v) => s + v, 0) / arr.length
  let V = 0
  for (const v of arr) V += (v - m) ** 2
  return V / arr.length
}

function sampleChain(act: Act, init: Init, L: number): number[] {
  const rng = mulberry32(2)
  const scales: Record<Init, number> = {
    naive: 1, // unit variance, no fan-in scaling
    xavier: Math.sqrt(1 / WIDTH),
    he: Math.sqrt(2 / WIDTH),
  }
  const scale = scales[init]

  // Build a single batch of N_SAMPLES activations, all initially N(0, 1).
  let a: number[][] = []
  for (let i = 0; i < N_SAMPLES; i++) {
    const row: number[] = []
    for (let j = 0; j < WIDTH; j++) row.push(gauss(rng))
    a.push(row)
  }

  const variances: number[] = [variance(a.flat())]

  for (let layer = 0; layer < L; layer++) {
    // Build a single weight matrix (WIDTH, WIDTH)
    const W: number[][] = []
    for (let i = 0; i < WIDTH; i++) {
      const row: number[] = []
      for (let j = 0; j < WIDTH; j++) row.push(gauss(rng) * scale)
      W.push(row)
    }
    const next: number[][] = []
    for (let s = 0; s < N_SAMPLES; s++) {
      const out: number[] = []
      for (let i = 0; i < WIDTH; i++) {
        let z = 0
        for (let j = 0; j < WIDTH; j++) z += W[i][j] * a[s][j]
        out.push(act === 'relu' ? Math.max(0, z) : Math.tanh(z))
      }
      next.push(out)
    }
    a = next
    variances.push(variance(a.flat()))
  }
  return variances
}

export default function ActivationVarianceChain() {
  const [activation, setActivation] = useState<Act>('relu')
  const [L, setL] = useState(20)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const series = useMemo(
    () => ({
      naive: sampleChain(activation, 'naive', MAX_L),
      xavier: sampleChain(activation, 'xavier', MAX_L),
      he: sampleChain(activation, 'he', MAX_L),
    }),
    [activation],
  )

  const colors: Record<Init, string> = {
    naive: '#f87171',
    xavier: '#60a5fa',
    he: '#fbbf24',
  }

  useEffect(() => {
    const canvas = canvasRef.current
    const box = boxRef.current
    if (!canvas || !box) return
    const dpr = window.devicePixelRatio || 1

    const draw = () => {
      const w = box.clientWidth
      const h = box.clientHeight
      canvas.width = w * dpr
      canvas.height = h * dpr
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      const ctx = canvas.getContext('2d')!
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, w, h)

      const padL = 56
      const padR = 12
      const padT = 18
      const padB = 32
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const toSx = (k: number) => padL + (k / MAX_L) * plotW
      const toSy = (v: number) => {
        const log = Math.log10(Math.max(v, 1e-14))
        const yMax = 6
        const yMin = -12
        return padT + plotH - ((log - yMin) / (yMax - yMin)) * plotH
      }

      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1

      ctx.textAlign = 'right'
      ;[6, 3, 0, -3, -6, -9].forEach((p) => {
        const sy = toSy(Math.pow(10, p))
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(p === 0 ? '1' : `10${toSup(p)}`, padL - 6, sy + 3)
      })
      // Highlight "variance = 1" — the ideal
      ctx.strokeStyle = '#4ade80'
      ctx.setLineDash([4, 4])
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(padL, toSy(1))
      ctx.lineTo(padL + plotW, toSy(1))
      ctx.stroke()
      ctx.setLineDash([])

      ctx.textAlign = 'center'
      ;[0, 10, 20, 30].forEach((k) => ctx.fillText(String(k), toSx(k), padT + plotH + 14))
      ctx.fillStyle = '#777'
      ctx.fillText('layer depth k', padL + plotW / 2, padT + plotH + 26)

      // Curves
      ;(['naive', 'xavier', 'he'] as Init[]).forEach((init) => {
        ctx.strokeStyle = colors[init]
        ctx.lineWidth = 2
        ctx.beginPath()
        series[init].slice(0, L + 1).forEach((v, k) => {
          const sx = toSx(k)
          const sy = toSy(v)
          if (k === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        ctx.stroke()
        // Faded remainder
        ctx.globalAlpha = 0.25
        ctx.setLineDash([3, 3])
        ctx.beginPath()
        series[init].slice(L).forEach((v, idx) => {
          const sx = toSx(L + idx)
          const sy = toSy(v)
          if (idx === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        ctx.stroke()
        ctx.setLineDash([])
        ctx.globalAlpha = 1
      })

      // Legend
      let lx = padL + 8
      const ly = padT + 14
      ;(['naive', 'xavier', 'he'] as Init[]).forEach((init) => {
        ctx.fillStyle = colors[init]
        ctx.fillRect(lx, ly - 6, 10, 2)
        ctx.fillStyle = '#ccc'
        ctx.textAlign = 'left'
        const label = init === 'naive' ? 'naive (σ=1)' : `${init} init`
        ctx.fillText(label, lx + 14, ly)
        lx += 14 + ctx.measureText(label).width + 18
      })
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [series, L])

  const finalValues = (['naive', 'xavier', 'he'] as Init[]).map((init) => ({
    init,
    v: series[init][L],
  }))

  return (
    <WidgetFrame
      widgetName="ActivationVarianceChain"
      label="activation variance through depth"
      right={
        <>
          <span className="font-mono">width {WIDTH} · batch {N_SAMPLES}</span>
          <span className="text-dark-text-disabled">·</span>
          <span className="text-term-green">green dashed = Var = 1 (ideal)</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            {(['tanh', 'relu'] as Act[]).map((a) => (
              <button
                key={a}
                onClick={() => setActivation(a)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
                  activation === a
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
                )}
              >
                {a}
              </button>
            ))}
          </div>
          <Slider
            label="depth L"
            value={L}
            min={1}
            max={MAX_L}
            step={1}
            onChange={setL}
            format={(v) => String(Math.round(v))}
            accent="accent-dark-accent"
          />
          <div className="flex items-center gap-4 ml-auto">
            {finalValues.map(({ init, v }) => (
              <Readout
                key={init}
                label={init}
                value={fmt(v)}
                accent={
                  init === 'naive' ? 'text-term-rose' : init === 'xavier' ? 'text-dark-sql' : 'text-term-amber'
                }
              />
            ))}
          </div>
        </div>
      }
    >
      <div ref={boxRef} className="absolute inset-0">
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>
    </WidgetFrame>
  )
}

function fmt(v: number): string {
  if (v === 0) return '0'
  if (v < 1e-3 || v > 1e4) return v.toExponential(1)
  return v.toFixed(3)
}

function toSup(n: number): string {
  const map = '⁰¹²³⁴⁵⁶⁷⁸⁹'
  return String(n)
    .split('')
    .map((c) => (c === '-' ? '⁻' : map[Number(c)] ?? c))
    .join('')
}
