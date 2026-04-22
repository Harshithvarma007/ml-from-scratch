'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Button, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'
import { StepForward, RotateCcw } from 'lucide-react'

// A 4-state × 4-action Q-table. Pressing "step" walks through a fixed training
// tape (s, a, r, s') and performs one Bellman update:
//   Q(s,a) ← Q(s,a) + α·(r + γ·max_{a'} Q(s',a') − Q(s,a)).
// The cell being touched glows; the formula below plugs in the actual numbers
// so the math is reproducible by hand.

const STATES = ['s0', 's1', 's2', 's3']
const ACTIONS = ['a0', 'a1', 'a2', 'a3']

type Transition = { s: number; a: number; r: number; sp: number }

const TAPE: Transition[] = [
  { s: 0, a: 1, r: 0, sp: 1 },
  { s: 1, a: 2, r: 0, sp: 2 },
  { s: 2, a: 0, r: 0, sp: 3 },
  { s: 3, a: 3, r: 10, sp: 3 },
  { s: 0, a: 2, r: 0, sp: 2 },
  { s: 2, a: 3, r: 0, sp: 3 },
  { s: 3, a: 3, r: 10, sp: 3 },
  { s: 0, a: 1, r: 0, sp: 1 },
  { s: 1, a: 3, r: -1, sp: 1 },
  { s: 1, a: 2, r: 0, sp: 2 },
  { s: 2, a: 3, r: 0, sp: 3 },
  { s: 3, a: 3, r: 10, sp: 3 },
]

function makeInitialQ(): number[][] {
  return STATES.map(() => ACTIONS.map(() => 0))
}

function maxQ(Q: number[][], s: number): number {
  return Math.max(...Q[s])
}

function cellColor(q: number, absMax: number): string {
  if (absMax < 1e-6) return 'rgba(30, 30, 30, 0.6)'
  const t = q / absMax
  if (t >= 0) return `rgba(74, 222, 128, ${0.12 + Math.min(1, t) * 0.6})`
  return `rgba(251, 113, 133, ${0.12 + Math.min(1, -t) * 0.6})`
}

export default function QTableUpdate() {
  const [alpha, setAlpha] = useState(0.3)
  const [gamma, setGamma] = useState(0.9)
  const [stepIdx, setStepIdx] = useState(0)
  const [Q, setQ] = useState<number[][]>(makeInitialQ)

  const cur = TAPE[stepIdx % TAPE.length]
  const q_sa = Q[cur.s][cur.a]
  const target = cur.r + gamma * maxQ(Q, cur.sp)
  const tdError = target - q_sa
  const newQ = q_sa + alpha * tdError

  const absMax = useMemo(() => {
    let m = 0
    for (let s = 0; s < STATES.length; s++)
      for (let a = 0; a < ACTIONS.length; a++) m = Math.max(m, Math.abs(Q[s][a]))
    return Math.max(m, 1)
  }, [Q])

  const doStep = () => {
    const next = Q.map((row) => row.slice())
    next[cur.s][cur.a] = newQ
    setQ(next)
    setStepIdx(stepIdx + 1)
  }

  const reset = () => {
    setQ(makeInitialQ())
    setStepIdx(0)
  }

  return (
    <WidgetFrame
      widgetName="QTableUpdate"
      label="Q-table update — one Bellman step at a time"
      right={
        <span className="font-mono">4 states · 4 actions · tape length {TAPE.length}</span>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="α"
            value={alpha}
            min={0}
            max={1}
            step={0.01}
            onChange={setAlpha}
            format={(v) => v.toFixed(2)}
            accent="accent-term-amber"
          />
          <Slider
            label="γ"
            value={gamma}
            min={0}
            max={0.999}
            step={0.001}
            onChange={setGamma}
            format={(v) => v.toFixed(3)}
            accent="accent-term-cyan"
          />
          <div className="flex items-center gap-1.5">
            <Button onClick={doStep} variant="primary">
              <span className="inline-flex items-center gap-1">
                step <StepForward size={11} />
              </span>
            </Button>
            <Button onClick={reset}>
              <span className="inline-flex items-center gap-1">
                <RotateCcw size={11} /> reset
              </span>
            </Button>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="step" value={`${stepIdx} / ${TAPE.length - 1}`} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[1fr_340px] gap-5 overflow-hidden">
        {/* Q-table */}
        <div className="flex flex-col min-h-0 min-w-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
            Q(s, a) — green = higher, rose = lower
          </div>
          <div
            className="grid gap-1.5"
            style={{
              gridTemplateColumns: `56px repeat(${ACTIONS.length}, 1fr)`,
              gridTemplateRows: `auto repeat(${STATES.length}, 1fr)`,
            }}
          >
            <div />
            {ACTIONS.map((a) => (
              <div
                key={a}
                className="text-center text-[10.5px] font-mono text-dark-text-secondary"
              >
                {a}
              </div>
            ))}
            {STATES.map((s, si) => (
              <div key={s} className="contents">
                <div className="flex items-center text-[10.5px] font-mono text-dark-text-secondary">
                  {s}
                </div>
                {ACTIONS.map((a, ai) => {
                  const highlighted = si === cur.s && ai === cur.a
                  const nextStateCol = si === cur.sp
                  return (
                    <div
                      key={a}
                      className={cn(
                        'relative rounded border flex items-center justify-center font-mono text-[12px] tabular-nums py-3 transition-all',
                        highlighted
                          ? 'border-term-amber ring-1 ring-term-amber shadow-[0_0_12px_rgba(251,191,36,0.4)]'
                          : nextStateCol
                          ? 'border-term-cyan/40'
                          : 'border-dark-border',
                      )}
                      style={{ backgroundColor: cellColor(Q[si][ai], absMax) }}
                    >
                      <span className="text-dark-text-primary">
                        {Q[si][ai].toFixed(2)}
                      </span>
                      {highlighted && (
                        <span className="absolute -top-2 left-1 text-[8.5px] px-1 rounded-sm bg-term-amber text-dark-bg font-semibold">
                          update
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>
            ))}
          </div>

          <div className="mt-3 text-[10.5px] font-mono text-dark-text-muted leading-snug">
            tape[{stepIdx % TAPE.length}] = <span className="text-term-amber">s={STATES[cur.s]}</span>,
            <span className="text-term-cyan"> a={ACTIONS[cur.a]}</span>,
            <span className={cn(cur.r > 0 ? 'text-term-green' : cur.r < 0 ? 'text-term-rose' : 'text-dark-text-muted')}>
              {' '}r={cur.r >= 0 ? '+' : ''}{cur.r}
            </span>,
            <span className="text-term-amber"> s'={STATES[cur.sp]}</span>
          </div>
        </div>

        {/* Math panel */}
        <div className="flex flex-col gap-3 min-w-0 min-h-0 overflow-auto">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            Bellman update — plugged in
          </div>
          <div className="rounded border border-term-amber/40 bg-term-amber/5 p-3 font-mono text-[11px] leading-relaxed space-y-1.5">
            <div className="text-dark-text-muted">
              Q(s,a) ← Q(s,a) + α·(r + γ·max Q(s',·) − Q(s,a))
            </div>
            <div className="border-t border-dark-border my-2" />
            <Row k="Q(s,a)" v={q_sa.toFixed(3)} c="text-dark-text-primary" />
            <Row k="r" v={cur.r.toFixed(2)} c={cur.r > 0 ? 'text-term-green' : cur.r < 0 ? 'text-term-rose' : 'text-dark-text-muted'} />
            <Row k="γ" v={gamma.toFixed(3)} c="text-term-cyan" />
            <Row k="max Q(s',·)" v={maxQ(Q, cur.sp).toFixed(3)} c="text-term-amber" />
            <Row k="target = r + γ·max" v={target.toFixed(3)} c="text-term-amber" />
            <Row k="TD error δ" v={tdError.toFixed(3)} c={tdError > 0 ? 'text-term-green' : 'text-term-rose'} />
            <Row k="α·δ" v={(alpha * tdError).toFixed(3)} c="text-term-purple" />
            <div className="border-t border-dark-border my-2" />
            <Row k="Q_new(s,a)" v={newQ.toFixed(3)} c="text-term-green" bold />
          </div>

          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-2">
            intuition
          </div>
          <p className="font-sans text-[12px] text-dark-text-muted leading-relaxed">
            α controls how much of the TD error we absorb into Q. γ controls how far the bootstrap peeks ahead.
            Watch the (s,a) cell drift toward the best neighbor of s'.
          </p>
        </div>
      </div>
    </WidgetFrame>
  )
}

function Row({
  k,
  v,
  c,
  bold,
}: {
  k: string
  v: string
  c: string
  bold?: boolean
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-dark-text-secondary">{k}</span>
      <span className={cn('tabular-nums', c, bold && 'font-semibold')}>{v}</span>
    </div>
  )
}
