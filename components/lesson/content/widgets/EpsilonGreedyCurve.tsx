'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Three ε-decay schedules on one canvas. constant stays flat; linear drops
// on a straight line to ε_final; exponential half-lives down at rate κ. A
// second mini-plot shows "exploration fraction" — for each strategy, what
// percentage of actions taken by t are random. Move the cursor (click / drag)
// to inspect any step. Sliders tune ε_final and the exponential rate.

const STEPS = 1000

type Strategy = {
  key: 'const' | 'linear' | 'exp'
  label: string
  color: string
}

const STRATEGIES: Strategy[] = [
  { key: 'const', label: 'constant', color: '#f472b6' },
  { key: 'linear', label: 'linear', color: '#67e8f9' },
  { key: 'exp', label: 'exponential', color: '#fbbf24' },
]

function curve(strategy: Strategy['key'], t: number, epsFinal: number, rate: number): number {
  const epsStart = 1.0
  const f = t / (STEPS - 1)
  if (strategy === 'const') return 0.1
  if (strategy === 'linear') return Math.max(epsFinal, epsStart + (epsFinal - epsStart) * f)
  return Math.max(epsFinal, epsStart * Math.exp(-rate * t))
}

export default function EpsilonGreedyCurve() {
  const [epsFinal, setEpsFinal] = useState(0.05)
  const [rate, setRate] = useState(0.005)
  const [cursor, setCursor] = useState(STEPS - 1)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const miniRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const eps = useMemo(() => {
    const out: Record<Strategy['key'], number[]> = { const: [], linear: [], exp: [] }
    for (let t = 0; t < STEPS; t++) {
      out.const.push(curve('const', t, epsFinal, rate))
      out.linear.push(curve('linear', t, epsFinal, rate))
      out.exp.push(curve('exp', t, epsFinal, rate))
    }
    return out
  }, [epsFinal, rate])

  // Exploration fraction = running mean of ε
  const explFrac = useMemo(() => {
    const out: Record<Strategy['key'], number[]> = { const: [], linear: [], exp: [] }
    ;(['const', 'linear', 'exp'] as const).forEach((k) => {
      let sum = 0
      for (let t = 0; t < STEPS; t++) {
        sum += eps[k][t]
        out[k].push(sum / (t + 1))
      }
    })
    return out
  }, [eps])

  // Draw main plot
  useEffect(() => {
    const canvas = canvasRef.current
    const box = boxRef.current
    if (!canvas || !box) return
    const dpr = window.devicePixelRatio || 1
    const draw = () => {
      const w = box.clientWidth
      const h = Math.max(160, box.clientHeight * 0.62)
      canvas.width = w * dpr
      canvas.height = h * dpr
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      const ctx = canvas.getContext('2d')!
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, w, h)
      const padL = 40, padR = 16, padT = 14, padB = 22
      const plotW = w - padL - padR
      const plotH = h - padT - padB
      const toSx = (t: number) => padL + (t / (STEPS - 1)) * plotW
      const toSy = (v: number) => padT + plotH - v * plotH

      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.strokeStyle = '#1e1e1e'
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      ;[0, 0.25, 0.5, 0.75, 1].forEach((v) => {
        const sy = toSy(v)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(w - padR, sy)
        ctx.stroke()
        ctx.fillText(v.toFixed(2), padL - 6, sy + 3)
      })
      ctx.textAlign = 'center'
      ;[0, 250, 500, 750, 999].forEach((t) => ctx.fillText(String(t), toSx(t), padT + plotH + 14))

      STRATEGIES.forEach((s) => {
        ctx.strokeStyle = s.color
        ctx.lineWidth = 2
        ctx.beginPath()
        for (let t = 0; t < STEPS; t++) {
          const sx = toSx(t)
          const sy = toSy(eps[s.key][t])
          if (t === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        ctx.stroke()
      })

      const sxC = toSx(cursor)
      ctx.strokeStyle = 'rgba(255,255,255,0.35)'
      ctx.setLineDash([3, 4])
      ctx.beginPath()
      ctx.moveTo(sxC, padT)
      ctx.lineTo(sxC, padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])

      STRATEGIES.forEach((s) => {
        ctx.fillStyle = s.color
        ctx.beginPath()
        ctx.arc(sxC, toSy(eps[s.key][cursor]), 3.5, 0, Math.PI * 2)
        ctx.fill()
      })

      // Title strip
      ctx.fillStyle = '#888'
      ctx.textAlign = 'left'
      ctx.fillText('ε(t) — probability of exploring at step t', padL, padT - 3)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [eps, cursor])

  // Mini plot
  useEffect(() => {
    const canvas = miniRef.current
    const box = boxRef.current
    if (!canvas || !box) return
    const dpr = window.devicePixelRatio || 1
    const draw = () => {
      const w = box.clientWidth
      const h = Math.max(96, box.clientHeight * 0.34)
      canvas.width = w * dpr
      canvas.height = h * dpr
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      const ctx = canvas.getContext('2d')!
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, w, h)
      const padL = 40, padR = 16, padT = 14, padB = 16
      const plotW = w - padL - padR
      const plotH = h - padT - padB
      const toSx = (t: number) => padL + (t / (STEPS - 1)) * plotW
      const toSy = (v: number) => padT + plotH - v * plotH

      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.strokeStyle = '#1e1e1e'
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      ;[0, 0.5, 1].forEach((v) => {
        const sy = toSy(v)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(w - padR, sy)
        ctx.stroke()
        ctx.fillText(v.toFixed(1), padL - 6, sy + 3)
      })

      STRATEGIES.forEach((s) => {
        ctx.strokeStyle = s.color
        ctx.lineWidth = 1.8
        ctx.beginPath()
        for (let t = 0; t < STEPS; t++) {
          const sx = toSx(t)
          const sy = toSy(explFrac[s.key][t])
          if (t === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        ctx.stroke()
      })

      const sxC = toSx(cursor)
      ctx.strokeStyle = 'rgba(255,255,255,0.35)'
      ctx.setLineDash([3, 4])
      ctx.beginPath()
      ctx.moveTo(sxC, padT)
      ctx.lineTo(sxC, padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])

      ctx.fillStyle = '#888'
      ctx.textAlign = 'left'
      ctx.fillText('exploration fraction = mean ε up to step t', padL, padT - 3)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [explFrac, cursor])

  const handleMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const box = boxRef.current
    if (!box) return
    const rect = box.getBoundingClientRect()
    const frac = (e.clientX - rect.left) / rect.width
    setCursor(Math.max(0, Math.min(STEPS - 1, Math.round(frac * STEPS))))
  }

  return (
    <WidgetFrame
      widgetName="EpsilonGreedyCurve"
      label="ε-greedy — three decay schedules"
      right={<span className="font-mono">hover to inspect any step</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="ε_final"
            value={epsFinal}
            min={0}
            max={0.5}
            step={0.01}
            onChange={setEpsFinal}
            format={(v) => v.toFixed(2)}
            accent="accent-term-cyan"
          />
          <Slider
            label="exp rate κ"
            value={rate}
            min={0.001}
            max={0.03}
            step={0.0005}
            onChange={setRate}
            format={(v) => v.toFixed(4)}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="t" value={String(cursor)} />
            <Readout label="ε(const)" value={eps.const[cursor].toFixed(3)} accent="text-term-pink" />
            <Readout label="ε(lin)" value={eps.linear[cursor].toFixed(3)} accent="text-term-cyan" />
            <Readout label="ε(exp)" value={eps.exp[cursor].toFixed(3)} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <div
          ref={boxRef}
          className="w-full h-full flex flex-col gap-1 cursor-crosshair"
          onMouseMove={handleMove}
          onClick={handleMove}
        >
          <div className="flex items-center gap-4 text-[10.5px] font-mono">
            {STRATEGIES.map((s) => (
              <span key={s.key} className="inline-flex items-center gap-1.5">
                <span className={cn('inline-block w-3 h-[2px]')} style={{ backgroundColor: s.color }} />
                <span className="text-dark-text-secondary">{s.label}</span>
              </span>
            ))}
          </div>
          <canvas ref={canvasRef} className="w-full block" />
          <canvas ref={miniRef} className="w-full block" />
        </div>
      </div>
    </WidgetFrame>
  )
}
