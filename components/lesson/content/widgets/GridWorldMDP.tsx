'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'
import { RotateCcw, Play } from 'lucide-react'

// 5×5 grid world that makes the MDP tuple (s, a, r, s') concrete. The agent
// starts at the top-left, the goal sits at the bottom-right (+10), a pit at
// (2,3) zaps for -10, and two walls block free paths. Arrow keys or buttons
// drive the agent. Every step emits a trajectory row. The "solve" button runs
// value iteration and paints V(s) as a heatmap over the grid.

type Cell = 'empty' | 'wall' | 'goal' | 'pit' | 'start'

const ROWS = 5
const COLS = 5
const GAMMA = 0.9
const STEP_COST = -0.04

type Pos = { r: number; c: number }

const GRID: Cell[][] = [
  ['start', 'empty', 'empty', 'empty', 'empty'],
  ['empty', 'wall', 'empty', 'wall', 'empty'],
  ['empty', 'empty', 'empty', 'pit', 'empty'],
  ['empty', 'wall', 'empty', 'empty', 'empty'],
  ['empty', 'empty', 'empty', 'empty', 'goal'],
]

const START: Pos = { r: 0, c: 0 }

type Action = 'up' | 'down' | 'left' | 'right'
const ACTIONS: Action[] = ['up', 'down', 'left', 'right']
const ACTION_LABEL: Record<Action, string> = { up: '↑', down: '↓', left: '←', right: '→' }
const ACTION_DELTA: Record<Action, [number, number]> = {
  up: [-1, 0],
  down: [1, 0],
  left: [0, -1],
  right: [0, 1],
}

function cellAt(r: number, c: number): Cell {
  return GRID[r][c]
}

function step(pos: Pos, a: Action): { next: Pos; reward: number; terminal: boolean } {
  const [dr, dc] = ACTION_DELTA[a]
  const nr = pos.r + dr
  const nc = pos.c + dc
  // Out of bounds or wall — bounce back
  const blocked =
    nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS || cellAt(nr, nc) === 'wall'
  const next = blocked ? pos : { r: nr, c: nc }
  const kind = cellAt(next.r, next.c)
  if (kind === 'goal') return { next, reward: 10, terminal: true }
  if (kind === 'pit') return { next, reward: -10, terminal: true }
  return { next, reward: STEP_COST, terminal: false }
}

type TrajRow = { s: string; a: Action; r: number; sPrime: string }

function solveValueIteration(): number[][] {
  const V: number[][] = Array.from({ length: ROWS }, () => Array(COLS).fill(0))
  for (let iter = 0; iter < 200; iter++) {
    let maxDelta = 0
    const newV = V.map((row) => row.slice())
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const kind = cellAt(r, c)
        if (kind === 'wall') { newV[r][c] = 0; continue }
        if (kind === 'goal') { newV[r][c] = 10; continue }
        if (kind === 'pit') { newV[r][c] = -10; continue }
        let best = -Infinity
        for (const a of ACTIONS) {
          const { next, reward, terminal } = step({ r, c }, a)
          const future = terminal ? 0 : GAMMA * V[next.r][next.c]
          const q = reward + future
          if (q > best) best = q
        }
        newV[r][c] = best
        maxDelta = Math.max(maxDelta, Math.abs(best - V[r][c]))
      }
    }
    for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) V[r][c] = newV[r][c]
    if (maxDelta < 1e-4) break
  }
  return V
}

function posLabel(p: Pos): string {
  return `(${p.r},${p.c})`
}

function valueColor(v: number, vMin: number, vMax: number): string {
  const span = Math.max(Math.abs(vMin), Math.abs(vMax), 1e-6)
  const t = v / span
  if (t >= 0) return `rgba(74, 222, 128, ${0.1 + Math.min(1, t) * 0.55})`
  return `rgba(251, 113, 133, ${0.1 + Math.min(1, -t) * 0.55})`
}

export default function GridWorldMDP() {
  const [agent, setAgent] = useState<Pos>(START)
  const [lastAction, setLastAction] = useState<Action | null>(null)
  const [lastReward, setLastReward] = useState<number>(0)
  const [terminal, setTerminal] = useState(false)
  const [trajectory, setTrajectory] = useState<TrajRow[]>([])
  const [solved, setSolved] = useState(false)

  const V = useMemo(() => solveValueIteration(), [])
  const vFlat = V.flat().filter((_, i) => cellAt(Math.floor(i / COLS), i % COLS) !== 'wall')
  const vMin = Math.min(...vFlat)
  const vMax = Math.max(...vFlat)

  const take = useCallback(
    (a: Action) => {
      if (terminal) return
      const { next, reward, terminal: t } = step(agent, a)
      setTrajectory((prev) => [
        ...prev.slice(-14),
        { s: posLabel(agent), a, r: reward, sPrime: posLabel(next) },
      ])
      setLastAction(a)
      setLastReward(reward)
      setAgent(next)
      setTerminal(t)
    },
    [agent, terminal],
  )

  const reset = () => {
    setAgent(START)
    setLastAction(null)
    setLastReward(0)
    setTerminal(false)
    setTrajectory([])
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowUp') { e.preventDefault(); take('up') }
      else if (e.key === 'ArrowDown') { e.preventDefault(); take('down') }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); take('left') }
      else if (e.key === 'ArrowRight') { e.preventDefault(); take('right') }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [take])

  return (
    <WidgetFrame
      widgetName="GridWorldMDP"
      label="5×5 MDP — move the agent, read the tuple (s,a,r,s')"
      right={<span className="font-mono">γ = {GAMMA} · step cost = {STEP_COST}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <Button onClick={() => take('up')} disabled={terminal}>↑ up</Button>
            <Button onClick={() => take('down')} disabled={terminal}>↓ down</Button>
            <Button onClick={() => take('left')} disabled={terminal}>← left</Button>
            <Button onClick={() => take('right')} disabled={terminal}>→ right</Button>
            <Button onClick={() => setSolved((s) => !s)} variant="primary">
              <span className="inline-flex items-center gap-1">
                <Play size={11} /> {solved ? 'hide V(s)' : 'solve'}
              </span>
            </Button>
            <Button onClick={reset}>
              <span className="inline-flex items-center gap-1">
                <RotateCcw size={11} /> reset
              </span>
            </Button>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="last a" value={lastAction ? ACTION_LABEL[lastAction] : '—'} accent="text-term-cyan" />
            <Readout
              label="last r"
              value={lastReward.toFixed(2)}
              accent={lastReward > 0 ? 'text-term-green' : lastReward < -1 ? 'text-term-rose' : 'text-term-amber'}
            />
            <Readout label="state" value={posLabel(agent)} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[1fr_260px] gap-5 overflow-hidden">
        {/* Grid */}
        <div className="flex flex-col min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
            {solved ? 'V(s) heatmap — converged after value iteration' : 'grid world — arrow keys or buttons'}
          </div>
          <div
            className="grid gap-1 flex-1 min-h-0"
            style={{ gridTemplateColumns: `repeat(${COLS}, 1fr)`, gridTemplateRows: `repeat(${ROWS}, 1fr)` }}
          >
            {GRID.flatMap((row, r) =>
              row.map((kind, c) => {
                const isAgent = agent.r === r && agent.c === c
                const isStart = kind === 'start'
                const isGoal = kind === 'goal'
                const isPit = kind === 'pit'
                const isWall = kind === 'wall'
                const bg =
                  solved && !isWall
                    ? valueColor(V[r][c], vMin, vMax)
                    : isWall
                    ? '#1a1a1a'
                    : isGoal
                    ? 'rgba(74, 222, 128, 0.2)'
                    : isPit
                    ? 'rgba(251, 113, 133, 0.2)'
                    : '#0f0f1a'
                const border =
                  isAgent
                    ? 'border-term-amber'
                    : isGoal
                    ? 'border-term-green'
                    : isPit
                    ? 'border-term-rose'
                    : 'border-dark-border'
                return (
                  <div
                    key={`${r}-${c}`}
                    className={cn(
                      'relative rounded flex items-center justify-center font-mono border transition-colors',
                      border,
                      isAgent && 'ring-1 ring-term-amber',
                    )}
                    style={{ backgroundColor: bg }}
                  >
                    {isWall ? (
                      <span className="text-[10px] text-dark-text-disabled">wall</span>
                    ) : isGoal ? (
                      <span className="text-[10px] text-term-green">+10</span>
                    ) : isPit ? (
                      <span className="text-[10px] text-term-rose">-10</span>
                    ) : isStart && !isAgent ? (
                      <span className="text-[9px] text-dark-text-muted">S</span>
                    ) : null}
                    {isAgent && (
                      <div className="absolute inset-2 rounded-full bg-term-amber/80 shadow-[0_0_12px_rgba(251,191,36,0.8)]" />
                    )}
                    {solved && !isWall && !isGoal && !isPit && (
                      <span className="absolute bottom-1 right-1 text-[8.5px] tabular-nums text-dark-text-primary">
                        {V[r][c].toFixed(1)}
                      </span>
                    )}
                  </div>
                )
              }),
            )}
          </div>
          {terminal && (
            <div className="mt-2 text-[11px] font-mono text-term-amber">
              terminal reached — press reset to start a new episode.
            </div>
          )}
        </div>

        {/* Sidebar: trajectory log */}
        <div className="flex flex-col min-h-0 min-w-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2">
            trajectory · (s, a, r, s')
          </div>
          <div className="flex-1 min-h-0 overflow-auto rounded border border-dark-border bg-dark-bg font-mono text-[10.5px]">
            {trajectory.length === 0 ? (
              <div className="p-3 text-dark-text-muted">no steps yet — take an action.</div>
            ) : (
              <ul className="divide-y divide-dark-border">
                {trajectory.map((row, i) => (
                  <li key={i} className="px-2.5 py-1 flex items-center gap-2">
                    <span className="text-dark-text-disabled w-5 text-right">{i + 1}</span>
                    <span className="text-term-amber tabular-nums">{row.s}</span>
                    <span className="text-term-cyan">{ACTION_LABEL[row.a]}</span>
                    <span
                      className={cn(
                        'tabular-nums',
                        row.r > 0
                          ? 'text-term-green'
                          : row.r < -1
                          ? 'text-term-rose'
                          : 'text-dark-text-muted',
                      )}
                    >
                      {row.r > 0 ? '+' : ''}
                      {row.r.toFixed(2)}
                    </span>
                    <span className="text-dark-text-disabled">→</span>
                    <span className="text-term-amber tabular-nums">{row.sPrime}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
