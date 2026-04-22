'use client'

// 200-token document shown as chip strip. A sliding window of context_len
// moves across, highlighting the current input span; the cell right after the
// window is the target token. Slider for context_len (16/64/128) and stride.
// Auto-play advances the window.

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { Play, Pause, StepForward } from 'lucide-react'
import { cn } from '@/lib/utils'

const N = 200

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Fake token strings — just varied enough to look like text
function buildDoc(): string[] {
  const rng = mulberry32(7)
  const fragments = [
    'the', '▁and', '▁of', 'ing', 'ed', '▁a', '▁to', '▁is', '▁it', 'ly',
    'tion', '▁for', '▁with', '▁by', '▁on', 'er', 'al', '▁was', '▁are', 'ness',
    '▁in', '▁at', '▁this', '▁that', '▁not', 'able', 'ment', '▁have', '▁be', '▁we',
  ]
  const out: string[] = []
  for (let i = 0; i < N; i++) {
    out.push(fragments[Math.floor(rng() * fragments.length)])
  }
  return out
}

const CTX_OPTIONS = [16, 64, 128]

export default function ContextWindowViz() {
  const [ctxLen, setCtxLen] = useState(64)
  const [stride, setStride] = useState(32)
  const [start, setStart] = useState(0)
  const [playing, setPlaying] = useState(false)

  const doc = useMemo(buildDoc, [])

  // Valid start positions: 0, stride, 2*stride, ... such that start + ctxLen < N
  const lastStart = Math.max(0, N - ctxLen - 1)
  useEffect(() => {
    if (start > lastStart) setStart(0)
  }, [lastStart, start])

  const rafRef = useRef<number | null>(null)
  const accRef = useRef(0)
  const lastTsRef = useRef<number | null>(null)
  useEffect(() => {
    if (!playing) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
      lastTsRef.current = null
      return
    }
    const tick = (ts: number) => {
      if (lastTsRef.current == null) lastTsRef.current = ts
      const dt = (ts - lastTsRef.current) / 1000
      lastTsRef.current = ts
      accRef.current += dt
      if (accRef.current > 0.45) {
        accRef.current = 0
        setStart((s) => {
          const next = s + stride
          if (next > lastStart) return 0
          return next
        })
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
      lastTsRef.current = null
    }
  }, [playing, stride, lastStart])

  const windowEnd = Math.min(N - 1, start + ctxLen - 1)
  const targetIdx = Math.min(N - 1, start + ctxLen)
  const targetTok = doc[targetIdx]
  const numSamples = Math.floor(lastStart / stride) + 1

  const stepManually = () => {
    setStart((s) => {
      const next = s + stride
      return next > lastStart ? 0 : next
    })
  }

  return (
    <WidgetFrame
      widgetName="ContextWindowViz"
      label="sliding context window — next-token targets"
      right={
        <span className="font-mono">
          doc length {N} · {numSamples} training samples @ stride {stride}
        </span>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <Button onClick={() => setPlaying(!playing)} variant="primary">
              <span className="inline-flex items-center gap-1">
                {playing ? <Pause size={11} /> : <Play size={11} />}
                {playing ? 'pause' : 'auto'}
              </span>
            </Button>
            <Button onClick={stepManually} variant="ghost">
              <span className="inline-flex items-center gap-1">
                step <StepForward size={11} />
              </span>
            </Button>
          </div>
          <div className="flex items-center gap-1.5">
            {CTX_OPTIONS.map((v) => (
              <button
                key={v}
                onClick={() => {
                  setCtxLen(v)
                  if (stride > v) setStride(v)
                }}
                className={cn(
                  'px-2 py-1 rounded text-[10.5px] font-mono transition-all',
                  ctxLen === v
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                ctx {v}
              </button>
            ))}
          </div>
          <label className="flex items-center gap-3 flex-1 min-w-[220px] font-mono text-[12px]">
            <span className="text-dark-text-secondary whitespace-nowrap">stride</span>
            <input
              type="range"
              min={1}
              max={ctxLen}
              step={1}
              value={stride}
              onChange={(e) => setStride(Number(e.target.value))}
              className="flex-1 h-1 rounded-full bg-dark-border cursor-pointer accent-term-cyan"
            />
            <span className="text-dark-text-primary tabular-nums w-10 text-right">{stride}</span>
          </label>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="start" value={String(start)} />
            <Readout label="target" value={`t=${targetIdx}`} accent="text-term-pink" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 flex flex-col gap-3 overflow-hidden">
        <div className="flex items-center gap-2 text-[11px] font-mono">
          <span className="text-term-cyan">context</span>
          <span className="text-dark-text-secondary">
            = doc[{start} : {start + ctxLen}]
          </span>
          <span className="text-dark-text-disabled mx-1">·</span>
          <span className="text-term-pink">target</span>
          <span className="text-dark-text-secondary">= doc[{targetIdx}]</span>
          <span className="ml-auto text-dark-text-disabled">window = {ctxLen}</span>
        </div>

        {/* Doc strip */}
        <div className="flex-1 min-h-0 overflow-hidden">
          <div
            className="grid gap-[2px] p-2 bg-dark-surface-elevated/30 border border-dark-border rounded h-full content-start"
            style={{ gridTemplateColumns: `repeat(25, 1fr)` }}
          >
            {doc.map((tok, i) => {
              const inWindow = i >= start && i <= windowEnd
              const isTarget = i === targetIdx
              return (
                <div
                  key={i}
                  className={cn(
                    'h-auto aspect-[4/3] rounded-[2px] flex items-center justify-center text-[8px] font-mono border transition-colors',
                    isTarget
                      ? 'border-term-pink bg-term-pink/25 text-term-pink'
                      : inWindow
                        ? 'border-term-cyan/60 bg-term-cyan/15 text-term-cyan'
                        : 'border-dark-border/40 bg-dark-bg text-dark-text-disabled',
                  )}
                  title={`${i}: "${tok}"`}
                >
                  {tok.replace(/▁/g, '\u00b7').slice(0, 3)}
                </div>
              )
            })}
          </div>
        </div>

        {/* Target explanation bar */}
        <div className="flex items-center gap-3 border border-term-pink/30 rounded bg-term-pink/5 p-2 font-mono text-[11px]">
          <span className="text-term-pink">next-token target:</span>
          <span className="text-dark-text-primary">doc[{targetIdx}] = &quot;{targetTok.replace(/▁/g, '\u00b7')}&quot;</span>
          <span className="ml-auto text-dark-text-disabled">
            each sample = ({ctxLen} inputs → 1 target); shifted by {stride} for the next sample
          </span>
        </div>
      </div>
    </WidgetFrame>
  )
}
