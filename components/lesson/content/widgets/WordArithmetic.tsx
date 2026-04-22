'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// king - man + woman ≈ queen, rendered as a parallelogram in 2D. Pick one of
// four preset analogies or swap any of the three slots yourself; the widget
// resolves the resulting vector to its nearest word in the tiny vocabulary.

type Vec = readonly number[]

type Entry = {
  word: string
  // 2D position for the parallelogram drawing.
  xy: readonly [number, number]
  // 8-dim basis: [royalty, male, female, place, france, italy, motion, speed]
  vec: Vec
}

const VOCAB: Entry[] = [
  { word: 'king',    xy: [0.30, 0.30], vec: [1.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] },
  { word: 'queen',   xy: [0.60, 0.25], vec: [1.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0] },
  { word: 'prince',  xy: [0.26, 0.42], vec: [0.9, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] },
  { word: 'princess',xy: [0.58, 0.40], vec: [0.9, 0.0, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0] },

  { word: 'man',     xy: [0.22, 0.75], vec: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] },
  { word: 'woman',   xy: [0.54, 0.72], vec: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] },
  { word: 'boy',     xy: [0.18, 0.86], vec: [0.0, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] },
  { word: 'girl',    xy: [0.50, 0.86], vec: [0.0, 0.0, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0] },

  { word: 'paris',   xy: [0.78, 0.30], vec: [0.0, 0.0, 0.0, 1.0, 0.95, 0.0, 0.0, 0.0] },
  { word: 'france',  xy: [0.84, 0.45], vec: [0.0, 0.0, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0] },
  { word: 'rome',    xy: [0.82, 0.58], vec: [0.0, 0.0, 0.0, 1.0, 0.0, 0.95, 0.0, 0.0] },
  { word: 'italy',   xy: [0.90, 0.70], vec: [0.0, 0.0, 0.0, 0.6, 0.0, 1.0, 0.0, 0.0] },

  { word: 'walk',    xy: [0.10, 0.18], vec: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2] },
  { word: 'walking', xy: [0.14, 0.06], vec: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5] },
  { word: 'swim',    xy: [0.40, 0.18], vec: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.3] },
  { word: 'swimming',xy: [0.44, 0.06], vec: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.6] },

  { word: 'big',     xy: [0.92, 0.85], vec: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0] },
  { word: 'bigger',  xy: [0.96, 0.95], vec: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.4] },
  { word: 'small',   xy: [0.70, 0.92], vec: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] },
  { word: 'smaller', xy: [0.74, 0.98], vec: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4] },
]

type Preset = { a: string; b: string; c: string; expect: string }
const PRESETS: readonly Preset[] = [
  { a: 'king',    b: 'man',  c: 'woman', expect: 'queen' },
  { a: 'paris',   b: 'france', c: 'italy', expect: 'rome' },
  { a: 'walking', b: 'walk', c: 'swim',  expect: 'swimming' },
  { a: 'bigger',  b: 'big',  c: 'small', expect: 'smaller' },
] as const

function cosine(a: Vec, b: Vec): number {
  let dot = 0, an = 0, bn = 0
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; an += a[i] * a[i]; bn += b[i] * b[i] }
  const d = Math.sqrt(an) * Math.sqrt(bn)
  return d === 0 ? 0 : dot / d
}

function addSub(a: Vec, b: Vec, c: Vec): Vec {
  return a.map((_, i) => a[i] - b[i] + c[i])
}

function nearestWord(v: Vec, exclude: readonly string[]): { word: Entry; sim: number } {
  let best: { w: Entry; s: number } | null = null
  for (const e of VOCAB) {
    if (exclude.includes(e.word)) continue
    const s = cosine(v, e.vec)
    if (!best || s > best.s) best = { w: e, s }
  }
  // safe fallback
  const chosen = best ?? { w: VOCAB[0], s: 0 }
  return { word: chosen.w, sim: chosen.s }
}

// 2D position for the result: a - b + c in the display plane. This is purely
// for the parallelogram visualization — the actual nearest-word lookup uses
// the 8-dim vec above.
function addSubXY(a: readonly [number, number], b: readonly [number, number], c: readonly [number, number]): [number, number] {
  return [a[0] - b[0] + c[0], a[1] - b[1] + c[1]]
}

export default function WordArithmetic() {
  const [a, setA] = useState('king')
  const [b, setB] = useState('man')
  const [c, setC] = useState('woman')
  const [presetIdx, setPresetIdx] = useState<number | null>(0)

  const aE = VOCAB.find((w) => w.word === a) ?? VOCAB[0]
  const bE = VOCAB.find((w) => w.word === b) ?? VOCAB[0]
  const cE = VOCAB.find((w) => w.word === c) ?? VOCAB[0]

  const result = useMemo(() => addSub(aE.vec, bE.vec, cE.vec), [aE, bE, cE])
  const nearest = useMemo(() => nearestWord(result, [a, b, c]), [result, a, b, c])

  const resXY = addSubXY(aE.xy, bE.xy, cE.xy)
  const xClamp = Math.max(0.04, Math.min(0.96, resXY[0]))
  const yClamp = Math.max(0.04, Math.min(0.96, resXY[1]))

  const applyPreset = (p: Preset, i: number) => {
    setA(p.a); setB(p.b); setC(p.c); setPresetIdx(i)
  }

  // Parallelogram coords in pixel space (viewBox 1000 × 560).
  const toSx = (x: number) => 24 + x * 952
  const toSy = (y: number) => 16 + y * 528
  const pA  = [toSx(aE.xy[0]), toSy(aE.xy[1])]
  const pB  = [toSx(bE.xy[0]), toSy(bE.xy[1])]
  const pC  = [toSx(cE.xy[0]), toSy(cE.xy[1])]
  const pR  = [toSx(xClamp),   toSy(yClamp)]

  const correct = presetIdx !== null && nearest.word.word === PRESETS[presetIdx].expect

  return (
    <WidgetFrame
      widgetName="WordArithmetic"
      label="word arithmetic — a − b + c ≈ ?"
      right={<span className="font-mono">nearest word in 8-dim basis (royalty, gender, place, tense, magnitude)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-3">
          {PRESETS.map((p, i) => (
            <button
              key={i}
              onClick={() => applyPreset(p, i)}
              className={cn(
                'px-2 py-1 rounded text-[10.5px] font-mono uppercase transition-all',
                presetIdx === i
                  ? 'bg-dark-accent text-white'
                  : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
              )}
            >
              {p.a}−{p.b}+{p.c}
            </button>
          ))}
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="nearest" value={nearest.word.word} accent="text-term-green" />
            <Readout label="cos" value={nearest.sim.toFixed(3)} accent="text-term-cyan" />
            {presetIdx !== null && (
              <Readout
                label="vs expected"
                value={correct ? 'match' : `≠ ${PRESETS[presetIdx].expect}`}
                accent={correct ? 'text-term-green' : 'text-term-rose'}
              />
            )}
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[1fr_240px] gap-4 overflow-hidden">
        {/* Parallelogram view */}
        <div className="relative min-h-0">
          <svg viewBox="0 0 1000 560" preserveAspectRatio="none" className="w-full h-full">
            <rect x={12} y={8} width={976} height={544} fill="none" stroke="#1e1e1e" strokeWidth={1} rx={4} />
            {[0.25, 0.5, 0.75].map((f) => (
              <g key={f}>
                <line x1={12 + f * 976} y1={8} x2={12 + f * 976} y2={552} stroke="#141419" strokeWidth={1} />
                <line x1={12} y1={8 + f * 544} x2={988} y2={8 + f * 544} stroke="#141419" strokeWidth={1} />
              </g>
            ))}

            {/* Parallelogram edges */}
            <line x1={pA[0]} y1={pA[1]} x2={pB[0]} y2={pB[1]} stroke="#a78bfa" strokeWidth={1.5} strokeDasharray="4 3" opacity={0.7} />
            <line x1={pC[0]} y1={pC[1]} x2={pR[0]} y2={pR[1]} stroke="#a78bfa" strokeWidth={1.5} strokeDasharray="4 3" opacity={0.7} />
            <line x1={pA[0]} y1={pA[1]} x2={pC[0]} y2={pC[1]} stroke="#67e8f9" strokeWidth={1.2} opacity={0.45} />
            <line x1={pB[0]} y1={pB[1]} x2={pR[0]} y2={pR[1]} stroke="#67e8f9" strokeWidth={1.2} opacity={0.45} />

            {/* Result arrow */}
            <line x1={pA[0]} y1={pA[1]} x2={pR[0]} y2={pR[1]} stroke="#4ade80" strokeWidth={2} markerEnd="url(#arrow-green)" />

            {/* All vocab (ghosted) */}
            {VOCAB.map((w) => {
              const sx = toSx(w.xy[0])
              const sy = toSy(w.xy[1])
              const active = [a, b, c, nearest.word.word].includes(w.word)
              return (
                <g key={w.word}>
                  <circle cx={sx} cy={sy} r={active ? 5 : 3} fill={active ? '#fff' : '#555'} opacity={active ? 1 : 0.4} />
                  <text
                    x={sx + 7}
                    y={sy + 4}
                    fontSize="10"
                    fill={
                      w.word === a
                        ? '#67e8f9'
                        : w.word === b
                        ? '#f87171'
                        : w.word === c
                        ? '#a78bfa'
                        : w.word === nearest.word.word
                        ? '#4ade80'
                        : '#555'
                    }
                    fontFamily="JetBrains Mono, monospace"
                    fontWeight={active ? 600 : 400}
                  >
                    {w.word}
                  </text>
                </g>
              )
            })}

            {/* Highlight rings */}
            <circle cx={pA[0]} cy={pA[1]} r={11} fill="none" stroke="#67e8f9" strokeWidth={1.4} />
            <circle cx={pB[0]} cy={pB[1]} r={11} fill="none" stroke="#f87171" strokeWidth={1.4} />
            <circle cx={pC[0]} cy={pC[1]} r={11} fill="none" stroke="#a78bfa" strokeWidth={1.4} />
            <circle cx={pR[0]} cy={pR[1]} r={13} fill="none" stroke="#4ade80" strokeWidth={1.6} strokeDasharray="3 3" />

            <defs>
              <marker id="arrow-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="#4ade80" />
              </marker>
            </defs>
          </svg>
        </div>

        {/* Slot pickers */}
        <div className="flex flex-col gap-3 min-w-0">
          <SlotPicker label="a" color="#67e8f9" value={a} onChange={(w) => { setA(w); setPresetIdx(null) }} />
          <SlotPicker label="− b" color="#f87171" value={b} onChange={(w) => { setB(w); setPresetIdx(null) }} />
          <SlotPicker label="+ c" color="#a78bfa" value={c} onChange={(w) => { setC(w); setPresetIdx(null) }} />

          <div className="border-t border-dark-border my-1" />
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">result</div>
          <div className="bg-dark-surface-elevated/40 rounded p-2 font-mono text-[11px] leading-relaxed">
            <div className="text-dark-text-muted">
              <span className="text-term-cyan">{a}</span> − <span className="text-term-rose">{b}</span> + <span className="text-term-purple">{c}</span>
            </div>
            <div className="text-term-green text-[13px] mt-1">≈ {nearest.word.word}</div>
            <div className="text-dark-text-disabled text-[10px] mt-1">cos = {nearest.sim.toFixed(3)}</div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function SlotPicker({
  label,
  color,
  value,
  onChange,
}: {
  label: string
  color: string
  value: string
  onChange: (w: string) => void
}) {
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <div className="font-mono text-[10px] uppercase tracking-wider" style={{ color }}>
        {label}
      </div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-dark-surface-elevated border border-dark-border rounded px-2 py-1 font-mono text-[11px] text-dark-text-primary outline-none focus:border-dark-border-hover"
      >
        {VOCAB.map((w) => (
          <option key={w.word} value={w.word}>
            {w.word}
          </option>
        ))}
      </select>
    </div>
  )
}
