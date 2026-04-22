'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// A scripted 50-step run of a one-unit LSTM and a one-unit vanilla RNN on
// the same task: "remember the input signal at step 5 until step 49." The
// LSTM uses hand-set gates that mimic what a trained LSTM learns: open input
// gate at step 5, close it, keep forget gate near 1 all the way. The vanilla
// RNN uses a single tanh recurrence and its signal decays geometrically. The
// viewer watches c_t (LSTM) and h_t (RNN) as two line traces plus a heatmap.

const T = 50
const SIGNAL_STEP = 5
const SIGNAL_VALUE = 0.9

function runLSTM(forgetStrength: number, leakiness: number) {
  // forgetStrength: what f_t is when the network is in "hold" mode (ideally 1).
  // leakiness: optional small noise that leaks into the cell each step.
  const c: number[] = [0]
  const h: number[] = [0]
  const f: number[] = []
  const i: number[] = []
  const o: number[] = []
  for (let t = 0; t < T; t++) {
    const isWriteStep = t === SIGNAL_STEP
    const isReadStep = t === T - 1
    const ft = isWriteStep ? 0 : forgetStrength
    const it = isWriteStep ? 1 : 0
    const gt = isWriteStep ? SIGNAL_VALUE : 0
    const ot = isReadStep ? 1 : 0.1
    f.push(ft); i.push(it); o.push(ot)
    const ct = ft * c[t] + it * gt + leakiness * (Math.sin(t) - 0.5) * 0.02
    const ht = ot * Math.tanh(ct)
    c.push(ct)
    h.push(ht)
  }
  return { c: c.slice(1), h: h.slice(1), f, i, o }
}

function runRNN(sigma: number) {
  // Run a scalar RNN with tanh(sigma * h + x). Feed the signal at step 5
  // and see how fast the trace decays afterwards.
  const h: number[] = [0]
  for (let t = 0; t < T; t++) {
    const x = t === SIGNAL_STEP ? SIGNAL_VALUE : 0
    h.push(Math.tanh(sigma * h[t] + x))
  }
  return h.slice(1)
}

export default function CellStateTracker() {
  const [forget, setForget] = useState(0.99)
  const [rho, setRho] = useState(0.8)
  const [leak, setLeak] = useState(0)
  const [cursor, setCursor] = useState(T - 1)

  const lstm = useMemo(() => runLSTM(forget, leak), [forget, leak])
  const rnn = useMemo(() => runRNN(rho), [rho])

  const chart = renderChart(lstm.c, rnn, cursor)

  const lstmAtCursor = lstm.c[cursor]
  const rnnAtCursor = rnn[cursor]
  const lstmAtEnd = lstm.c[T - 1]
  const rnnAtEnd = rnn[T - 1]

  return (
    <WidgetFrame
      widgetName="CellStateTracker"
      label="cell state vs. hidden state — who remembers?"
      right={
        <span className="font-mono">
          signal injected at t = {SIGNAL_STEP} · {T} steps total
        </span>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider label="LSTM f_t" value={forget} min={0.5} max={1.0} step={0.005} onChange={setForget} format={(v) => v.toFixed(3)} accent="accent-term-amber" />
          <Slider label="RNN ρ" value={rho} min={0.3} max={1.2} step={0.01} onChange={setRho} format={(v) => v.toFixed(2)} accent="accent-term-rose" />
          <Slider label="leak" value={leak} min={0} max={1} step={0.05} onChange={setLeak} format={(v) => v.toFixed(2)} />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="LSTM c_T" value={lstmAtEnd.toFixed(3)} accent="text-term-amber" />
            <Readout label="RNN h_T" value={rnnAtEnd.toFixed(3)} accent={Math.abs(rnnAtEnd) < 0.05 ? 'text-term-rose' : 'text-term-green'} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 flex flex-col gap-2 overflow-hidden">
        {/* Chart */}
        <div className="relative flex-1 min-h-0">{chart}</div>

        {/* Gate ribbon */}
        <div className="space-y-1.5">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            LSTM gates over time
          </div>
          <GateRow label="f_t" values={lstm.f} color="#f87171" cursor={cursor} onScrub={setCursor} />
          <GateRow label="i_t" values={lstm.i} color="#67e8f9" cursor={cursor} onScrub={setCursor} />
          <GateRow label="o_t" values={lstm.o} color="#4ade80" cursor={cursor} onScrub={setCursor} />
        </div>

        {/* Cursor readout */}
        <div className="flex items-center gap-4 pt-1 font-mono text-[10.5px] text-dark-text-muted">
          <span className="text-dark-text-disabled uppercase tracking-wider text-[9.5px]">at cursor t = {cursor + 1}</span>
          <span>LSTM c_t = <span className="text-term-amber">{lstmAtCursor.toFixed(3)}</span></span>
          <span>RNN h_t = <span className={Math.abs(rnnAtCursor) < 0.05 ? 'text-term-rose' : 'text-term-green'}>{rnnAtCursor.toFixed(3)}</span></span>
        </div>
      </div>
    </WidgetFrame>
  )
}

function renderChart(lstmC: number[], rnnH: number[], cursor: number): JSX.Element {
  const W = 1000
  const H = 220
  const padL = 40
  const padR = 12
  const padT = 10
  const padB = 22
  const plotW = W - padL - padR
  const plotH = H - padT - padB
  const toSx = (t: number) => padL + (t / (T - 1)) * plotW
  const yMin = -1
  const yMax = 1
  const toSy = (v: number) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH

  const lstmPath = lstmC.map((v, t) => `${t === 0 ? 'M' : 'L'} ${toSx(t).toFixed(1)} ${toSy(v).toFixed(1)}`).join(' ')
  const rnnPath = rnnH.map((v, t) => `${t === 0 ? 'M' : 'L'} ${toSx(t).toFixed(1)} ${toSy(v).toFixed(1)}`).join(' ')

  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="w-full h-full">
      {/* Grid */}
      {[-1, -0.5, 0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1={padL} y1={toSy(v)} x2={W - padR} y2={toSy(v)} stroke="#1e1e1e" strokeWidth={1} />
          <text x={padL - 6} y={toSy(v) + 3} textAnchor="end" fontSize="9" fill="#555" fontFamily="JetBrains Mono, monospace">
            {v.toFixed(1)}
          </text>
        </g>
      ))}
      {[0, 10, 20, 30, 40, 49].map((t) => (
        <text key={t} x={toSx(t)} y={H - 6} textAnchor="middle" fontSize="9" fill="#555" fontFamily="JetBrains Mono, monospace">
          {t}
        </text>
      ))}

      {/* Signal marker at step 5 */}
      <line x1={toSx(SIGNAL_STEP)} y1={padT} x2={toSx(SIGNAL_STEP)} y2={padT + plotH} stroke="rgba(167,139,250,0.5)" strokeDasharray="3 3" />
      <text x={toSx(SIGNAL_STEP) + 4} y={padT + 11} fontSize="9" fill="#a78bfa" fontFamily="JetBrains Mono, monospace">
        signal
      </text>

      {/* Traces */}
      <path d={rnnPath} stroke="#f87171" strokeWidth={2} fill="none" />
      <path d={lstmPath} stroke="#fbbf24" strokeWidth={2} fill="none" />

      {/* Cursor dot */}
      <line x1={toSx(cursor)} y1={padT} x2={toSx(cursor)} y2={padT + plotH} stroke="rgba(255,255,255,0.3)" strokeDasharray="2 3" />
      <circle cx={toSx(cursor)} cy={toSy(lstmC[cursor])} r={4} fill="#fbbf24" />
      <circle cx={toSx(cursor)} cy={toSy(rnnH[cursor])} r={4} fill="#f87171" />

      {/* Legend */}
      <g transform={`translate(${padL + 8}, ${padT + 4})`}>
        <line x1={0} y1={8} x2={14} y2={8} stroke="#fbbf24" strokeWidth={2} />
        <text x={18} y={11} fontSize="10" fill="#ccc" fontFamily="JetBrains Mono, monospace">LSTM c_t</text>
        <line x1={90} y1={8} x2={104} y2={8} stroke="#f87171" strokeWidth={2} />
        <text x={108} y={11} fontSize="10" fill="#ccc" fontFamily="JetBrains Mono, monospace">RNN h_t</text>
      </g>
    </svg>
  )
}

function GateRow({
  label,
  values,
  color,
  cursor,
  onScrub,
}: {
  label: string
  values: number[]
  color: string
  cursor: number
  onScrub: (t: number) => void
}) {
  return (
    <div className="flex items-center gap-2 font-mono text-[10px]">
      <span className="w-6 text-dark-text-secondary">{label}</span>
      <div
        className="flex-1 grid gap-[1px]"
        style={{ gridTemplateColumns: `repeat(${T}, 1fr)` }}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect()
          const frac = (e.clientX - r.left) / r.width
          onScrub(Math.max(0, Math.min(T - 1, Math.floor(frac * T))))
        }}
      >
        {values.map((v, t) => (
          <div
            key={t}
            className={cn('h-3 rounded-[1px]', cursor === t && 'outline outline-1 outline-white/60')}
            style={{
              backgroundColor: color,
              opacity: 0.18 + Math.min(1, Math.abs(v)) * 0.75,
            }}
            title={`${label}[${t}] = ${v.toFixed(2)}`}
          />
        ))}
      </div>
    </div>
  )
}
