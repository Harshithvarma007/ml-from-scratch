'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { Play, Pause, SkipBack } from 'lucide-react'

// DDPM generation trajectory: start from pure noise x_T and run the reverse
// process for T ∈ {10, 50, 100} steps. Since we don't actually have a trained
// net, we simulate the oracle reverse update (DDIM-style) using a known
// target x_0 (a geometric scene). We render 10 evenly-spaced intermediate
// 32x32 tiles as a horizontal timeline, plus a cursor that tracks the
// current step, with a readout of variance (1 - alpha_bar_t).

const N = 32
const STEP_OPTS = [10, 50, 100]
const BETA_START = 1e-4
const BETA_END = 0.02

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

function targetImage(): number[] {
  // Two filled shapes + a bar — matches the noise-staircase target style.
  const out = new Array(N * N).fill(-0.6)
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++) {
      if (Math.hypot(x - 10, y - 11) < 6) out[y * N + x] = 0.85
      if (y > 17 && y < 27 && x > 19 && x < 28) out[y * N + x] = 0.55
    }
  return out
}

function buildTrajectory(numSteps: number, seed: number): { frames: number[][]; alphaBars: number[] } {
  // Build a schedule with `numSteps` total points (t=0..numSteps-1).
  const betas: number[] = []
  const alphaBar: number[] = []
  let prod = 1
  for (let t = 0; t < numSteps; t++) {
    const b = BETA_START + (BETA_END - BETA_START) * (t / Math.max(1, numSteps - 1))
    betas.push(b)
    prod *= 1 - b
    alphaBar.push(prod)
  }
  const x0 = targetImage()

  // sample x_T
  const rng = mulberry32(seed)
  const x_T = new Array(N * N)
  const ab_T = alphaBar[numSteps - 1]
  for (let i = 0; i < N * N; i++) {
    x_T[i] = Math.sqrt(ab_T) * x0[i] + Math.sqrt(1 - ab_T) * gauss(rng)
  }

  // frames[numSteps] = x_T; frames[0] = x_0
  const frames: number[][] = new Array(numSteps + 1)
  frames[numSteps] = x_T
  let cur = x_T
  for (let t = numSteps - 1; t >= 0; t--) {
    const ab_t = alphaBar[t]
    const ab_prev = t === 0 ? 1 : alphaBar[t - 1]
    const next = new Array(N * N)
    for (let i = 0; i < N * N; i++) {
      const eps = (cur[i] - Math.sqrt(ab_t) * x0[i]) / Math.max(Math.sqrt(1 - ab_t), 1e-6)
      next[i] = Math.sqrt(ab_prev) * x0[i] + Math.sqrt(1 - ab_prev) * eps
    }
    frames[t] = next
    cur = next
  }
  return { frames, alphaBars: alphaBar }
}

function toGray(v: number): string {
  const g = Math.max(0, Math.min(255, Math.round((v * 0.5 + 0.5) * 255)))
  return `rgb(${g},${g},${g})`
}

export default function DDPMGeneration() {
  const [stepChoice, setStepChoice] = useState<number>(50)
  const [seed, setSeed] = useState<number>(2024)
  const [cursor, setCursor] = useState<number>(stepChoice)
  const [playing, setPlaying] = useState(false)
  const rafRef = useRef<number | null>(null)

  const { frames, alphaBars } = useMemo(
    () => buildTrajectory(stepChoice, seed),
    [stepChoice, seed],
  )

  // 10 evenly spaced preview tiles along the trajectory (0 → stepChoice).
  const previewIndices = useMemo(() => {
    const arr: number[] = []
    for (let i = 0; i < 10; i++) arr.push(Math.round((i / 9) * stepChoice))
    return arr
  }, [stepChoice])

  // Reset cursor when step count changes
  useEffect(() => {
    setCursor(stepChoice)
    setPlaying(false)
  }, [stepChoice, seed])

  // Play loop
  useEffect(() => {
    if (!playing) return
    const tick = () => {
      setCursor((prev) => {
        if (prev <= 0) {
          setPlaying(false)
          return 0
        }
        return prev - 1
      })
      rafRef.current = window.setTimeout(tick, 60) as unknown as number
    }
    rafRef.current = window.setTimeout(tick, 60) as unknown as number
    return () => {
      if (rafRef.current) window.clearTimeout(rafRef.current)
    }
  }, [playing])

  const currentFrame = frames[Math.min(cursor, stepChoice)] ?? frames[0]
  const ab_now = alphaBars[Math.min(cursor, stepChoice - 1)]
  const var_now = 1 - ab_now

  return (
    <WidgetFrame
      widgetName="DDPMGeneration"
      label="DDPM sampling — noise re-condenses into an image"
      right={<span className="font-mono">x_T → x_0 · {stepChoice} reverse steps</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={() => { setCursor(stepChoice); setPlaying(false) }} variant="ghost">
            <span className="inline-flex items-center gap-1"><SkipBack size={11} /> reset</span>
          </Button>
          <Button onClick={() => setPlaying((p) => !p)} variant="primary" disabled={cursor === 0 && !playing}>
            <span className="inline-flex items-center gap-1">
              {playing ? (<><Pause size={11} /> pause</>) : (<><Play size={11} /> play</>)}
            </span>
          </Button>
          <Button onClick={() => setSeed((s) => s + 1)} variant="ghost">
            new noise
          </Button>
          <div className="flex items-center gap-1">
            {STEP_OPTS.map((s) => (
              <button
                key={s}
                onClick={() => setStepChoice(s)}
                className={`px-2 py-1 rounded text-[10.5px] font-mono transition-all ${
                  stepChoice === s
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
                }`}
              >
                T = {s}
              </button>
            ))}
          </div>
          <input
            type="range"
            min={0}
            max={stepChoice}
            step={1}
            value={cursor}
            onChange={(e) => { setCursor(Number(e.target.value)); setPlaying(false) }}
            className="flex-1 min-w-[100px] h-1 rounded-full bg-dark-border accent-term-amber cursor-pointer"
          />
          <div className="flex items-center gap-3 ml-auto">
            <Readout label="t" value={String(cursor)} accent="text-term-amber" />
            <Readout label="var" value={var_now.toFixed(3)} accent="text-term-cyan" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden flex flex-col gap-3">
        <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
          preview — 10 frames along the trajectory
        </div>
        <div className="flex items-center justify-between gap-1.5">
          {previewIndices.map((idx, i) => {
            const isCursor = idx === Math.min(cursor, stepChoice)
            const data = frames[idx] ?? frames[0]
            return (
              <div key={i} className="flex flex-col items-center gap-1 flex-1 min-w-0">
                <div
                  className={`text-[9px] font-mono tabular-nums ${isCursor ? 'text-term-amber' : 'text-dark-text-disabled'}`}
                >
                  t={idx}
                </div>
                <button
                  onClick={() => { setCursor(idx); setPlaying(false) }}
                  className={`aspect-square w-full max-w-[64px] rounded-sm border overflow-hidden transition-all ${
                    isCursor ? 'border-term-amber shadow-[0_0_0_1px_rgba(251,191,36,0.5)]' : 'border-dark-border'
                  }`}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: `repeat(${N}, 1fr)`,
                    gridTemplateRows: `repeat(${N}, 1fr)`,
                  }}
                >
                  {data.map((v, i2) => (
                    <div key={i2} style={{ backgroundColor: toGray(v) }} />
                  ))}
                </button>
              </div>
            )
          })}
        </div>

        <div className="flex items-center justify-center gap-4 pt-1">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            current x_{cursor}
          </div>
          <div
            className="aspect-square w-[140px] rounded border border-term-amber overflow-hidden"
            style={{
              display: 'grid',
              gridTemplateColumns: `repeat(${N}, 1fr)`,
              gridTemplateRows: `repeat(${N}, 1fr)`,
              boxShadow: '0 0 0 2px rgba(251,191,36,0.25)',
            }}
          >
            {currentFrame.map((v, i) => (
              <div key={i} style={{ backgroundColor: toGray(v) }} />
            ))}
          </div>
          <div className="text-[10.5px] font-mono text-dark-text-muted leading-relaxed">
            <div><span className="text-term-rose">x_T</span> = pure noise</div>
            <div><span className="text-term-green">x_0</span> = clean sample</div>
            <div className="mt-1">reverse step: <span className="text-term-amber">x_(t−1) ← denoise(x_t)</span></div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
