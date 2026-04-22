'use client'

import { useEffect, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout, Button } from './WidgetFrame'
import { Play, Pause, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'

// A cartoon simulation of what happens during training when you do (or don't)
// add the auxiliary load-balancing loss. Without LB loss the router gets
// positive feedback — the expert that happened to win first keeps winning,
// and the distribution collapses onto 1-2 experts. With LB loss, a gradient
// pushes router probabilities back toward uniform.
//
// We simulate a batch of 200 tokens per iteration for 50 iterations. Each
// iteration updates the router's per-expert bias b_i via two competing
// forces: (1) a "winners win" drift toward experts that saw more tokens,
// and (2) a load-balancing pull toward uniform, scaled by alpha.

const NUM_EXPERTS = 8
const NUM_TOKENS = 200
const MAX_ITER = 50

type State = {
  iter: number
  biases: number[]           // router per-expert bias b_i
  counts: number[]           // token counts per expert this iteration
  runningMean: number[]      // running mean of counts (for animation stability)
  ginis: number[]            // Gini-ish unevenness over iterations
  history: number[][]        // counts per iteration
}

function init(): State {
  // Tiny random initial bias so some expert wins the first round.
  const rng = mulberry32(7)
  return {
    iter: 0,
    biases: Array.from({ length: NUM_EXPERTS }, () => (rng() - 0.5) * 0.2),
    counts: new Array(NUM_EXPERTS).fill(0),
    runningMean: new Array(NUM_EXPERTS).fill(NUM_TOKENS / NUM_EXPERTS),
    ginis: [],
    history: [],
  }
}

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function softmax(xs: number[]): number[] {
  const m = Math.max(...xs)
  const es = xs.map((x) => Math.exp(x - m))
  const s = es.reduce((a, b) => a + b, 0)
  return es.map((e) => e / s)
}

function gini(xs: number[]): number {
  // A crude unevenness measure in [0, 1]: 0 uniform, ~1 collapsed.
  const s = xs.reduce((a, b) => a + b, 0) || 1
  const n = xs.length
  const sorted = [...xs].sort((a, b) => a - b)
  let num = 0
  for (let i = 0; i < n; i++) num += (2 * (i + 1) - n - 1) * sorted[i]
  return num / (n * s)
}

function step(state: State, alpha: number, rng: () => number): State {
  const biases = [...state.biases]
  const counts = new Array(NUM_EXPERTS).fill(0)
  for (let t = 0; t < NUM_TOKENS; t++) {
    // Give each token a weak random context so the distribution isn't totally deterministic.
    const ctx = Array.from({ length: NUM_EXPERTS }, () => (rng() - 0.5) * 0.4)
    const logits = biases.map((b, i) => b + ctx[i])
    const p = softmax(logits)
    // Top-1 routing for this simulation
    let bestI = 0
    let bestV = -Infinity
    for (let i = 0; i < NUM_EXPERTS; i++) if (p[i] > bestV) { bestV = p[i]; bestI = i }
    counts[bestI] += 1
  }

  // Update biases. Two terms:
  //   positive feedback: b_i += η · (count_i / N - 1/E)
  //   LB pull:           b_i -= η · α · (f_i − 1/E)
  // Net effect: positive feedback pushes toward the winner; LB pulls back.
  const eta = 0.08
  const uniform = 1 / NUM_EXPERTS
  for (let i = 0; i < NUM_EXPERTS; i++) {
    const f = counts[i] / NUM_TOKENS
    biases[i] += eta * (f - uniform) * (1 - alpha) // feedback, damped by α
    biases[i] -= eta * alpha * (f - uniform) * 3 // LB pulls ~3× stronger when active
  }

  const g = gini(counts)
  return {
    iter: state.iter + 1,
    biases,
    counts,
    runningMean: state.runningMean.map((m, i) => m * 0.6 + counts[i] * 0.4),
    ginis: [...state.ginis, g],
    history: [...state.history, counts],
  }
}

export default function ExpertCollapse() {
  const [alpha, setAlpha] = useState(0)
  const [state, setState] = useState<State>(init)
  const [playing, setPlaying] = useState(false)
  const rngRef = useRef(mulberry32(123))
  const rafRef = useRef<number | null>(null)
  const lastTick = useRef<number>(0)

  useEffect(() => {
    if (!playing) {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      return
    }
    const loop = (t: number) => {
      if (t - lastTick.current > 160) {
        lastTick.current = t
        setState((s) => {
          if (s.iter >= MAX_ITER) {
            setPlaying(false)
            return s
          }
          return step(s, alpha, rngRef.current)
        })
      }
      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    }
  }, [playing, alpha])

  const reset = () => {
    rngRef.current = mulberry32(123)
    setState(init())
    setPlaying(false)
  }

  const counts = state.counts.length ? state.counts : new Array(NUM_EXPERTS).fill(NUM_TOKENS / NUM_EXPERTS)
  const maxCount = Math.max(...counts, NUM_TOKENS / NUM_EXPERTS)
  const uniform = NUM_TOKENS / NUM_EXPERTS
  const currGini = state.ginis.length ? state.ginis[state.ginis.length - 1] : 0
  const collapsed = currGini > 0.45

  return (
    <WidgetFrame
      widgetName="ExpertCollapse"
      label="expert collapse — without LB loss, winners take all"
      right={
        <span className="font-mono">
          {NUM_TOKENS} tokens / iter · top-1 routing · iter {state.iter}/{MAX_ITER}
        </span>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="α (LB weight)"
            value={alpha}
            min={0}
            max={1}
            step={0.05}
            onChange={setAlpha}
            format={(v) => v.toFixed(2)}
            accent="accent-term-green"
          />
          <Button
            onClick={() => {
              if (state.iter >= MAX_ITER) reset()
              setPlaying((p) => !p)
            }}
            variant="primary"
          >
            <span className="inline-flex items-center gap-1">
              {playing ? <><Pause size={11} /> pause</> : <><Play size={11} /> play</>}
            </span>
          </Button>
          <Button onClick={reset} variant="ghost">
            <span className="inline-flex items-center gap-1">
              <RotateCcw size={11} /> reset
            </span>
          </Button>
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="unevenness"
              value={currGini.toFixed(2)}
              accent={collapsed ? 'text-term-rose' : 'text-term-green'}
            />
            <Readout label="iter" value={`${state.iter}/${MAX_ITER}`} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-1 md:grid-cols-[1fr_220px] gap-4">
        {/* Bar chart */}
        <div className="flex flex-col gap-2 min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            tokens routed per expert  ·  dashed line = uniform target
          </div>
          <div className="relative flex-1 flex items-end gap-2 bg-dark-bg rounded p-2">
            {/* Uniform target line */}
            <div
              className="absolute left-2 right-2 border-t border-dashed border-term-green/50"
              style={{ bottom: `${(uniform / maxCount) * 100}%` }}
            />
            <div
              className="absolute right-3 text-[9px] font-mono text-term-green/70"
              style={{ bottom: `calc(${(uniform / maxCount) * 100}% + 2px)` }}
            >
              uniform ({uniform.toFixed(0)})
            </div>
            {counts.map((c, i) => {
              const pct = (c / maxCount) * 100
              const overloaded = c > uniform * 1.5
              return (
                <div key={i} className="flex-1 flex flex-col items-center gap-1 h-full justify-end">
                  <div className="font-mono text-[9.5px] tabular-nums text-dark-text-muted">{c}</div>
                  <div
                    className="w-full rounded-t transition-all"
                    style={{
                      height: `${pct}%`,
                      backgroundColor: overloaded ? '#fb7185' : collapsed ? '#f472b6' : '#4ade80',
                      opacity: 0.85,
                    }}
                  />
                  <div className="font-mono text-[9.5px] text-dark-text-secondary">E{i}</div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Right column: unevenness trace + narrative */}
        <div className="flex flex-col gap-3 min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            unevenness over iters
          </div>
          <svg viewBox="0 0 200 90" className="w-full h-24 bg-dark-bg rounded">
            <line x1={10} y1={80} x2={195} y2={80} stroke="#333" strokeWidth={1} />
            <line x1={10} y1={10} x2={10} y2={80} stroke="#333" strokeWidth={1} />
            <text x={3} y={14} fontSize="7" fill="#555" fontFamily="JetBrains Mono, monospace">1</text>
            <text x={3} y={82} fontSize="7" fill="#555" fontFamily="JetBrains Mono, monospace">0</text>
            {/* Threshold */}
            <line
              x1={10}
              y1={80 - 0.45 * 70}
              x2={195}
              y2={80 - 0.45 * 70}
              stroke="#fb7185"
              strokeDasharray="2 2"
              strokeWidth={1}
              opacity={0.5}
            />
            <path
              d={state.ginis
                .map((g, i) => {
                  const x = 10 + (i / Math.max(MAX_ITER - 1, 1)) * 185
                  const y = 80 - g * 70
                  return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`
                })
                .join(' ')}
              stroke={collapsed ? '#fb7185' : '#4ade80'}
              strokeWidth={1.5}
              fill="none"
            />
          </svg>

          <div className="bg-dark-surface-elevated/40 rounded p-3 font-mono text-[10.5px] leading-relaxed">
            {alpha < 0.05 ? (
              <>
                <div className="text-term-rose mb-1">α ≈ 0 · no LB loss</div>
                <div className="text-dark-text-muted">
                  positive feedback wins. whichever expert gets the first extra token gets more next
                  round, and so on — 1-2 experts take everything.
                </div>
              </>
            ) : alpha < 0.4 ? (
              <>
                <div className="text-term-amber mb-1">α = {alpha.toFixed(2)} · mild LB</div>
                <div className="text-dark-text-muted">
                  some balancing, but the feedback loop still tilts the distribution. partial collapse.
                </div>
              </>
            ) : (
              <>
                <div className="text-term-green mb-1">α = {alpha.toFixed(2)} · strong LB</div>
                <div className="text-dark-text-muted">
                  LB gradient dominates the positive feedback. every expert gets roughly N/E tokens.
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
