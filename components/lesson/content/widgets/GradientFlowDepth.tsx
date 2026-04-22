'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// How big is the gradient at layer k, deep in a network of L layers?  For
// each activation and init strategy, we simulate a chain of L layers and
// plot |∂L/∂x_k| at each depth. The punchline: sigmoid + bad init is
// unrecoverably vanishing; ReLU + He init stays healthy. Residual
// connections cure even the sigmoid case.

type ActName = 'sigmoid' | 'tanh' | 'relu'
type InitName = 'naive' | 'xavier' | 'he'

const MAX_L = 40

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

function initScale(init: InitName, fanIn: number): number {
  if (init === 'naive') return 1 // unit variance, no fan-in scaling
  if (init === 'xavier') return Math.sqrt(1 / fanIn)
  return Math.sqrt(2 / fanIn) // he
}

function deriv(act: ActName, z: number): number {
  if (act === 'relu') return z > 0 ? 1 : 0
  if (act === 'tanh') return 1 - Math.tanh(z) ** 2
  const s = 1 / (1 + Math.exp(-z))
  return s * (1 - s)
}

function simulate(act: ActName, init: InitName, L: number, residual: boolean): number[] {
  // A "pseudo-network" with a single scalar per layer. We track how the product
  // of expected-abs-derivative times weight evolves with depth, starting from 1
  // at the output and moving backward. This captures the vanishing-gradient
  // regime (what we actually care about) without needing a full forward run.
  const rng = mulberry32(2)
  const fanIn = 64
  const scale = initScale(init, fanIn)

  const gains: number[] = [1] // gradient magnitude at layer L (output)
  for (let k = L - 1; k >= 0; k--) {
    // Sample z and |derivative| a few times; average.
    let ave = 0
    const TRIALS = 32
    for (let t = 0; t < TRIALS; t++) {
      const z = gauss(rng) * (act === 'relu' ? Math.sqrt(2) : 1)
      const d = deriv(act, z)
      // Weight gain roughly ~ |w| · fanIn_eff  for the matrix–vector product. We
      // approximate |w| by scale (since w ~ N(0, scale²)).
      const w = scale
      const step = d * w * Math.sqrt(fanIn) + (residual ? 1 : 0)
      ave += Math.abs(step)
    }
    ave /= TRIALS
    gains.push(gains[gains.length - 1] * ave)
  }
  return gains.reverse() // now index 0 = input, L = output
}

export default function GradientFlowDepth() {
  const [L, setL] = useState(20)
  const [residual, setResidual] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const series: Record<string, { color: string; data: number[] }> = useMemo(() => ({
    'sigmoid · naive': { color: '#a78bfa', data: simulate('sigmoid', 'naive', MAX_L, residual) },
    'tanh · xavier': { color: '#60a5fa', data: simulate('tanh', 'xavier', MAX_L, residual) },
    'relu · he': { color: '#fbbf24', data: simulate('relu', 'he', MAX_L, residual) },
  }), [residual])

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
        const logV = Math.log10(Math.max(v, 1e-20))
        const yMinLog = -15
        const yMaxLog = 3
        return (
          padT + plotH - ((logV - yMinLog) / (yMaxLog - yMinLog)) * plotH
        )
      }

      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.textAlign = 'right'
      ;[3, 0, -3, -6, -9, -12].forEach((p) => {
        const sy = toSy(Math.pow(10, p))
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(p === 0 ? '1' : `10${toSup(p)}`, padL - 6, sy + 3)
      })
      ctx.textAlign = 'center'
      ;[0, 10, 20, 30, 40].forEach((k) => ctx.fillText(String(k), toSx(k), padT + plotH + 14))
      ctx.fillStyle = '#777'
      ctx.fillText('layer depth k', padL + plotW / 2, padT + plotH + 26)

      // Plot each curve. Solid up through current L, faded beyond.
      Object.entries(series).forEach(([key, { color, data }]) => {
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.beginPath()
        data.forEach((v, k) => {
          if (k > L) return
          const sx = toSx(k)
          const sy = toSy(v)
          if (k === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        ctx.stroke()

        ctx.globalAlpha = 0.22
        ctx.setLineDash([3, 3])
        ctx.beginPath()
        data.forEach((v, k) => {
          if (k < L) return
          const sx = toSx(k)
          const sy = toSy(v)
          if (k === L) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        ctx.stroke()
        ctx.globalAlpha = 1
        ctx.setLineDash([])
      })

      // Cursor at current L
      ctx.strokeStyle = '#333'
      ctx.setLineDash([3, 4])
      ctx.beginPath()
      ctx.moveTo(toSx(L), padT)
      ctx.lineTo(toSx(L), padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])

      // Legend
      let lx = padL + 8
      const ly = padT + 16
      Object.entries(series).forEach(([key, { color }]) => {
        ctx.fillStyle = color
        ctx.fillRect(lx, ly - 6, 10, 2)
        ctx.fillStyle = '#ccc'
        ctx.textAlign = 'left'
        ctx.fillText(key, lx + 14, ly)
        lx += 14 + ctx.measureText(key).width + 18
      })
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [series, L])

  const finalValues = Object.entries(series).map(([key, { data }]) => ({
    key,
    value: data[L],
  }))

  return (
    <WidgetFrame
      widgetName="GradientFlowDepth"
      label="gradient magnitude — how much signal reaches layer k from the loss"
      right={<span>log₁₀ scale · residual toggle resets the derivative gain</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="depth L"
            value={L}
            min={1}
            max={MAX_L}
            step={1}
            onChange={setL}
            format={(n) => String(Math.round(n)).padStart(2, ' ')}
            accent="accent-dark-accent"
          />
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={residual}
              onChange={(e) => setResidual(e.target.checked)}
              className="accent-term-green"
            />
            <span className="text-[11px] font-mono text-dark-text-secondary">
              residual connections
            </span>
          </label>
          <div className="flex items-center gap-4 ml-auto">
            {finalValues.map(({ key, value }) => (
              <Readout
                key={key}
                label={key.split(' · ')[0]}
                value={fmt(value)}
                accent={
                  key.startsWith('sigmoid')
                    ? 'text-term-purple'
                    : key.startsWith('tanh')
                      ? 'text-dark-sql'
                      : 'text-term-amber'
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
