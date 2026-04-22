'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Interactive sandbox: apply temperature → top-k → top-p IN SEQUENCE. Each
// stage is visualized as its own bar chart so you can watch the shape evolve.
// Readouts for entropy, effective vocabulary, and remaining probability mass.

const TOKENS = ['king', 'queen', 'prince', 'knight', 'the', 'and', 'of', 'a', 'to', 'in']
const LOGITS = [4.2, 3.6, 2.9, 2.4, 1.9, 1.4, 0.8, 0.2, -0.5, -1.3]

function softmax(x: number[]): number[] {
  const m = Math.max(...x.filter((v) => Number.isFinite(v)))
  const e = x.map((v) => (Number.isFinite(v) ? Math.exp(v - m) : 0))
  const s = e.reduce((a, b) => a + b, 0) || 1
  return e.map((v) => v / s)
}

function entropy(p: number[]): number {
  return -p.reduce((a, v) => a + (v > 1e-12 ? v * Math.log(v) : 0), 0)
}

function effectiveVocab(p: number[]): number {
  return p.reduce((a, v) => a + (v > 1e-6 ? 1 : 0), 0)
}

function applyTemperature(logits: number[], T: number): number[] {
  return logits.map((v) => v / Math.max(T, 1e-6))
}

function applyTopK(logits: number[], k: number): number[] {
  const sorted = logits.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v)
  const keep = new Set(sorted.slice(0, k).map((x) => x.i))
  return logits.map((v, i) => (keep.has(i) ? v : -Infinity))
}

function applyTopP(logits: number[], p: number): number[] {
  const probs = softmax(logits)
  const sorted = probs.map((pr, i) => ({ pr, i })).sort((a, b) => b.pr - a.pr)
  let cum = 0
  const keep = new Set<number>()
  for (const { pr, i } of sorted) {
    keep.add(i)
    cum += pr
    if (cum >= p) break
  }
  return logits.map((v, i) => (keep.has(i) ? v : -Infinity))
}

export default function DistributionShaper() {
  const [T, setT] = useState(1.0)
  const [K, setK] = useState(8)
  const [P, setP] = useState(0.9)

  const stage0P = useMemo(() => softmax(LOGITS), [])
  const stage1Logits = useMemo(() => applyTemperature(LOGITS, T), [T])
  const stage1P = useMemo(() => softmax(stage1Logits), [stage1Logits])

  const stage2Logits = useMemo(() => applyTopK(stage1Logits, K), [stage1Logits, K])
  const stage2P = useMemo(() => softmax(stage2Logits), [stage2Logits])

  const stage3Logits = useMemo(() => applyTopP(stage2Logits, P), [stage2Logits, P])
  const stage3P = useMemo(() => softmax(stage3Logits), [stage3Logits])

  const stages = [
    { name: 'raw', desc: 'logits → softmax', color: '#67e8f9', p: stage0P, logits: LOGITS },
    { name: `÷ T=${T.toFixed(2)}`, desc: 'divide logits by temperature', color: '#fbbf24', p: stage1P, logits: stage1Logits },
    { name: `top-k=${K}`, desc: `mask all but top ${K} logits`, color: '#a78bfa', p: stage2P, logits: stage2Logits },
    { name: `top-p=${P.toFixed(2)}`, desc: `keep smallest set with Σ ≥ ${P.toFixed(2)}`, color: '#f472b6', p: stage3P, logits: stage3Logits },
  ]

  return (
    <WidgetFrame
      widgetName="DistributionShaper"
      label="distribution shaper — compose temperature, top-k, top-p"
      right={<span className="font-mono">pipeline: logits → /T → top-k → top-p → softmax</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider label="T" value={T} min={0.1} max={2.0} step={0.05} onChange={setT} format={(v) => v.toFixed(2)} accent="accent-term-amber" />
          <Slider label="k" value={K} min={1} max={10} step={1} onChange={(v) => setK(Math.round(v))} format={(v) => String(Math.round(v))} accent="accent-term-purple" />
          <Slider label="p" value={P} min={0.1} max={1.0} step={0.05} onChange={setP} format={(v) => v.toFixed(2)} accent="accent-term-pink" />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="H(final)" value={entropy(stage3P).toFixed(3)} accent="text-term-pink" />
            <Readout label="eff vocab" value={String(effectiveVocab(stage3P))} accent="text-term-pink" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-4 gap-3 overflow-hidden">
        {stages.map((s, idx) => (
          <StageColumn key={idx} stage={s} idx={idx} />
        ))}
      </div>
    </WidgetFrame>
  )
}

function StageColumn({
  stage,
  idx,
}: {
  stage: { name: string; desc: string; color: string; p: number[]; logits: number[] }
  idx: number
}) {
  const H = entropy(stage.p)
  const vocab = effectiveVocab(stage.p)
  const pMass = stage.logits.reduce((a, v, i) => a + (Number.isFinite(v) ? stage.p[i] : 0), 0)

  return (
    <div
      className="flex flex-col gap-1 min-w-0 min-h-0 rounded border p-2"
      style={{ borderColor: 'rgba(63,63,70,0.7)' }}
    >
      <div className="flex items-center gap-2 font-mono text-[11px]">
        <span className="text-dark-text-disabled">stage {idx}</span>
        <span style={{ color: stage.color }} className="font-semibold truncate">
          {stage.name}
        </span>
      </div>
      <div className="text-[9.5px] font-mono text-dark-text-disabled truncate">{stage.desc}</div>
      <div className="grid grid-cols-3 gap-1 font-mono text-[9.5px] mt-1 mb-1 tabular-nums">
        <span className="text-dark-text-muted">
          H
          <br />
          <span className="text-dark-text-primary">{H.toFixed(2)}</span>
        </span>
        <span className="text-dark-text-muted">
          eff
          <br />
          <span className="text-dark-text-primary">{vocab}</span>
        </span>
        <span className="text-dark-text-muted">
          mass
          <br />
          <span className="text-dark-text-primary">{(pMass * 100).toFixed(0)}%</span>
        </span>
      </div>
      <div className="flex flex-col gap-0.5 flex-1 overflow-hidden">
        {TOKENS.map((t, i) => {
          const kept = Number.isFinite(stage.logits[i])
          return (
            <div key={i} className="flex items-center gap-1 font-mono text-[9px] min-w-0">
              <span className={kept ? 'w-11 text-dark-text-secondary truncate' : 'w-11 text-dark-text-disabled truncate'}>
                {t}
              </span>
              <div className="flex-1 h-2 bg-dark-bg rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${stage.p[i] * 100}%`,
                    backgroundColor: kept ? stage.color : '#3f3f46',
                    opacity: kept ? 0.85 : 0.25,
                  }}
                />
              </div>
              <span
                className={
                  kept
                    ? 'w-8 text-right tabular-nums text-dark-text-primary'
                    : 'w-8 text-right tabular-nums text-dark-text-disabled'
                }
              >
                {(stage.p[i] * 100).toFixed(0)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
