'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout, Button } from './WidgetFrame'
import { Play, Pause, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'

// A synthetic 6-layer network whose per-layer gradient norms are simulated
// as a function of (init-scale, activation). Plot them on a log-y axis. User
// drags init scale; if it's wrong for the chosen activation, curves fan out
// into exploding/vanishing regimes. Toggle between "ReLU + He" (healthy) and
// "sigmoid + naive" (vanishing) and "deep ReLU + unit init" (exploding).

type Regime = 'he-relu' | 'xavier-tanh' | 'naive-sigmoid' | 'unit-relu'

const regimes: Record<Regime, { label: string; kind: 'healthy' | 'vanish' | 'explode' }> = {
  'he-relu': { label: 'ReLU + He init', kind: 'healthy' },
  'xavier-tanh': { label: 'tanh + Xavier init', kind: 'healthy' },
  'naive-sigmoid': { label: 'sigmoid + σ=1', kind: 'vanish' },
  'unit-relu': { label: 'ReLU + σ=3', kind: 'explode' },
}

const LAYERS = 6
const MAX_STEP = 200

function mulberry32(seed: number) {
  let t = seed >>> 0
  return () => {
    t += 0x6d2b79f5
    let x = Math.imul(t ^ (t >>> 15), t | 1)
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61)
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296
  }
}

// Synthetic per-layer grad norm as a function of (layer, step, regime).
// The "healthy" regimes stay near 1. Vanish collapses toward 1e-9 at deep layers.
// Explode blows up toward 1e6.
function simulate(regime: Regime, step: number): number[] {
  const rng = mulberry32(7 + (regime === 'he-relu' ? 0 : regime === 'xavier-tanh' ? 1 : regime === 'naive-sigmoid' ? 2 : 3))
  const out: number[] = []
  for (let L = 0; L < LAYERS; L++) {
    // Base magnitude depends on regime + depth
    let base = 1.0
    if (regime === 'naive-sigmoid') base = Math.pow(0.2, LAYERS - 1 - L) // vanish toward earlier layers
    if (regime === 'unit-relu') base = Math.pow(2.0, LAYERS - 1 - L)
    // Add step-dependent noise + slow drift from the RNG
    const drift = 1 + 0.3 * Math.sin(step * 0.02 + L)
    const jitter = 0.5 + rng()
    out.push(Math.max(1e-14, base * drift * jitter))
  }
  return out
}

const COLORS = ['#67e8f9', '#a78bfa', '#fbbf24', '#4ade80', '#f472b6', '#f87171']

export default function GradNormMonitor() {
  const [regime, setRegime] = useState<Regime>('he-relu')
  const [step, setStep] = useState(40)
  const [playing, setPlaying] = useState(false)
  const rafRef = useRef<number | null>(null)
  const lastTickRef = useRef(0)
  const historyRef = useRef<number[][]>(Array.from({ length: LAYERS }, () => []))

  useEffect(() => {
    // Regenerate history when regime changes
    historyRef.current = Array.from({ length: LAYERS }, () => [])
    for (let s = 0; s <= step; s++) {
      const layerNorms = simulate(regime, s)
      for (let L = 0; L < LAYERS; L++) historyRef.current[L].push(layerNorms[L])
    }
  }, [regime])

  useEffect(() => {
    if (!playing) return
    const tick = (t: number) => {
      if (t - lastTickRef.current > 60) {
        lastTickRef.current = t
        setStep((s) => {
          if (s >= MAX_STEP) {
            setPlaying(false)
            return s
          }
          const nextLayerNorms = simulate(regime, s + 1)
          for (let L = 0; L < LAYERS; L++) historyRef.current[L].push(nextLayerNorms[L])
          return s + 1
        })
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [playing, regime])

  const reset = () => {
    setPlaying(false)
    setStep(40)
    historyRef.current = Array.from({ length: LAYERS }, () => [])
    for (let s = 0; s <= 40; s++) {
      const layerNorms = simulate(regime, s)
      for (let L = 0; L < LAYERS; L++) historyRef.current[L].push(layerNorms[L])
    }
  }

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

      const padL = 56
      const padR = 12
      const padT = 18
      const padB = 28
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const yMinLog = -12
      const yMaxLog = 6
      const toSx = (s: number) => padL + (s / MAX_STEP) * plotW
      const toSy = (v: number) => {
        const l = Math.log10(Math.max(v, 1e-14))
        return padT + plotH - ((l - yMinLog) / (yMaxLog - yMinLog)) * plotH
      }

      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.textAlign = 'right'
      ;[6, 3, 0, -3, -6, -9, -12].forEach((p) => {
        const sy = toSy(Math.pow(10, p))
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(p === 0 ? '1' : `10${toSup(p)}`, padL - 6, sy + 3)
      })
      ctx.textAlign = 'center'
      ;[0, 100, 200].forEach((s) => ctx.fillText(String(s), toSx(s), padT + plotH + 14))
      ctx.fillStyle = '#777'
      ctx.fillText('step', padL + plotW / 2, padT + plotH + 24)

      // Healthy band
      ctx.fillStyle = 'rgba(74, 222, 128, 0.04)'
      ctx.fillRect(padL, toSy(10), plotW, toSy(0.1) - toSy(10))

      for (let L = 0; L < LAYERS; L++) {
        const hist = historyRef.current[L]
        ctx.strokeStyle = COLORS[L]
        ctx.lineWidth = 1.5
        ctx.beginPath()
        hist.forEach((v, s) => {
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
        ctx.fillText(`L${L + 1}`, lx + 14, ly)
        lx += 38
      }
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [step, regime])

  const latestNorms = useMemo(
    () => historyRef.current.map((h) => h[h.length - 1] ?? 0),
    [step, regime],
  )
  const verdict = (() => {
    if (regime === 'naive-sigmoid') return 'vanishing · early layers frozen'
    if (regime === 'unit-relu') return 'exploding · training will diverge'
    return 'healthy · all layers receive gradient'
  })()
  const verdictColor =
    regime === 'naive-sigmoid' || regime === 'unit-relu'
      ? 'text-term-rose'
      : 'text-term-green'

  return (
    <WidgetFrame
      widgetName="GradNormMonitor"
      label="per-layer gradient norms — the diagnostic you'll use most"
      right={<span className="font-mono">log₁₀ scale · 6-layer net · green band = healthy</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1 flex-wrap">
            {(Object.keys(regimes) as Regime[]).map((r) => (
              <button
                key={r}
                onClick={() => {
                  setRegime(r)
                  setStep(40)
                }}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono uppercase transition-all',
                  regime === r
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
                )}
              >
                {regimes[r].label}
              </button>
            ))}
          </div>
          <Button onClick={() => setPlaying((p) => !p)} variant="primary">
            {playing ? <><Pause className="w-3 h-3 inline -mt-px mr-1" /> pause</> : <><Play className="w-3 h-3 inline -mt-px mr-1" /> play</>}
          </Button>
          <Button onClick={reset}>
            <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
          </Button>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="verdict" value={verdict} accent={verdictColor} />
            <Readout
              label="L1 / L6 ratio"
              value={
                latestNorms[0] && latestNorms[LAYERS - 1]
                  ? (latestNorms[0] / latestNorms[LAYERS - 1]).toExponential(1)
                  : '—'
              }
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

function toSup(n: number): string {
  const map = '⁰¹²³⁴⁵⁶⁷⁸⁹'
  return String(n)
    .split('')
    .map((c) => (c === '-' ? '⁻' : map[Number(c)] ?? c))
    .join('')
}
