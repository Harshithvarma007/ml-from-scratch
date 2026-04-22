'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Per-layer "% active" metric over a simulated training run. ReLU drifts
// downward if the LR is too high (neurons die). LeakyReLU stays flat. Slider
// for learning rate, toggle between ReLU and LeakyReLU.

type Act = 'relu' | 'leaky'

const LAYERS = 5
const STEPS = 80
const COLORS = ['#67e8f9', '#a78bfa', '#fbbf24', '#4ade80', '#f472b6']

function simulate(lr: number, act: Act): number[][] {
  // Start at 50% alive (fresh He init). Each step, a fraction of currently-
  // active neurons dies proportional to (lr - lr_safe)². Leaky never dies.
  const out: number[][] = Array.from({ length: LAYERS }, () => [])
  const alive = new Array(LAYERS).fill(0.5)
  const lrSafe = 0.5
  const kill = act === 'leaky' ? 0 : Math.max(0, Math.pow(lr - lrSafe, 2)) * 0.025
  for (let s = 0; s <= STEPS; s++) {
    for (let L = 0; L < LAYERS; L++) {
      // Deeper layers die faster (compounding)
      const factor = 1 + L * 0.3
      alive[L] = Math.max(0.01, alive[L] * (1 - kill * factor))
      out[L].push(alive[L])
    }
  }
  return out
}

export default function ActivationHealthChart() {
  const [lr, setLr] = useState(0.3)
  const [act, setAct] = useState<Act>('relu')
  const series = useMemo(() => simulate(lr, act), [lr, act])

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

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

      const padL = 48
      const padR = 12
      const padT = 20
      const padB = 30
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const toSx = (s: number) => padL + (s / STEPS) * plotW
      const toSy = (v: number) => padT + plotH - v * plotH

      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      ;[0, 0.25, 0.5, 0.75, 1].forEach((v) => {
        const sy = toSy(v)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(`${Math.round(v * 100)}%`, padL - 6, sy + 3)
      })
      ctx.textAlign = 'center'
      ;[0, 40, 80].forEach((s) => ctx.fillText(String(s), toSx(s), padT + plotH + 14))
      ctx.fillStyle = '#777'
      ctx.fillText('training step', padL + plotW / 2, padT + plotH + 24)

      for (let L = 0; L < LAYERS; L++) {
        ctx.strokeStyle = COLORS[L]
        ctx.lineWidth = 2
        ctx.beginPath()
        series[L].forEach((v, s) => {
          const sx = toSx(s)
          const sy = toSy(v)
          if (s === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        ctx.stroke()
      }

      // Legend
      let lx = padL + 8
      const ly = padT + 14
      for (let L = 0; L < LAYERS; L++) {
        ctx.fillStyle = COLORS[L]
        ctx.fillRect(lx, ly - 6, 10, 2)
        ctx.fillStyle = '#ccc'
        ctx.textAlign = 'left'
        ctx.fillText(`layer ${L + 1}`, lx + 14, ly)
        lx += 62
      }
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [series])

  const finalAlive = series.map((s) => s[s.length - 1])
  const avgAlive = finalAlive.reduce((a, b) => a + b, 0) / finalAlive.length
  const deepestAlive = finalAlive[LAYERS - 1]

  return (
    <WidgetFrame
      widgetName="ActivationHealthChart"
      label="% neurons alive — per layer, across training"
      right={<span className="font-mono">fresh init ≈ 50% active; watch deep layers fade</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            {(['relu', 'leaky'] as Act[]).map((a) => (
              <button
                key={a}
                onClick={() => setAct(a)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono uppercase transition-all',
                  act === a
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
                )}
              >
                {a === 'relu' ? 'ReLU' : 'Leaky ReLU'}
              </button>
            ))}
          </div>
          <Slider
            label="learning rate"
            value={lr}
            min={0.05}
            max={1.5}
            step={0.05}
            onChange={setLr}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="avg alive"
              value={`${(avgAlive * 100).toFixed(0)}%`}
              accent={avgAlive > 0.3 ? 'text-term-green' : 'text-term-rose'}
            />
            <Readout
              label="deepest layer"
              value={`${(deepestAlive * 100).toFixed(0)}%`}
              accent={deepestAlive > 0.2 ? 'text-term-green' : 'text-term-rose'}
            />
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
