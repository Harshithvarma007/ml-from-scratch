'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Vocabulary coverage from a synthetic Zipfian corpus. A slider picks a vocab
// size on a log scale and the widget reports % tokens covered plus shows the
// cutoff on the curve. A small bar chart of the top-10 frequent tokens
// anchors the intuition: a handful of words dominate, the tail is long.

const CORPUS_SIZE = 100_000
const VOCAB_N = 50_000

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Pre-compute Zipfian token counts with zeta parameter s, sum to CORPUS_SIZE.
function buildZipf(s: number): number[] {
  let Z = 0
  for (let k = 1; k <= VOCAB_N; k++) Z += 1 / Math.pow(k, s)
  const out: number[] = new Array(VOCAB_N)
  for (let k = 1; k <= VOCAB_N; k++) {
    out[k - 1] = (CORPUS_SIZE / Z) / Math.pow(k, s)
  }
  return out
}

// Pre-compute cumulative coverage fraction at each rank.
function cumulativeCoverage(counts: number[]): Float64Array {
  const total = counts.reduce((a, v) => a + v, 0)
  const cum = new Float64Array(counts.length)
  let running = 0
  for (let i = 0; i < counts.length; i++) {
    running += counts[i]
    cum[i] = running / total
  }
  return cum
}

// Picks reasonable top-10 token names from the corpus. Order is stable.
const TOP_WORDS = [
  'the', 'of', 'and', 'to', 'a', 'in', 'is', 'that', 'it', 'was',
]

function coverageAt(cum: Float64Array, vocabSize: number): number {
  const idx = Math.min(Math.max(1, Math.floor(vocabSize)), cum.length) - 1
  return cum[idx]
}

export default function VocabularyStats() {
  const [vocabLog, setVocabLog] = useState(3.3) // 10^3.3 ≈ 2000
  const [s, setS] = useState(1.07)

  const counts = useMemo(() => buildZipf(s), [s])
  const cum = useMemo(() => cumulativeCoverage(counts), [counts])

  const vocab = Math.round(Math.pow(10, vocabLog))
  const cov = coverageAt(cum, vocab)

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
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)

      const padL = 44
      const padR = 16
      const padT = 14
      const padB = 28
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const xMin = 0
      const xMax = Math.log10(VOCAB_N)
      const toSx = (log10V: number) => padL + ((log10V - xMin) / (xMax - xMin)) * plotW
      const toSy = (c: number) => padT + plotH - c * plotH

      // grid
      ctx.font = '9.5px "JetBrains Mono", monospace'
      ctx.strokeStyle = '#1a1a1e'
      ctx.fillStyle = '#555'
      ctx.lineWidth = 1
      for (const y of [0, 0.25, 0.5, 0.75, 1]) {
        ctx.beginPath()
        ctx.moveTo(padL, toSy(y))
        ctx.lineTo(padL + plotW, toSy(y))
        ctx.stroke()
        ctx.textAlign = 'right'
        ctx.fillText(`${(y * 100).toFixed(0)}%`, padL - 6, toSy(y) + 3)
      }
      for (const x of [1, 2, 3, 4]) {
        const sx = toSx(x)
        ctx.beginPath()
        ctx.moveTo(sx, padT)
        ctx.lineTo(sx, padT + plotH)
        ctx.stroke()
        ctx.textAlign = 'center'
        ctx.fillText(`10^${x}`, sx, padT + plotH + 13)
      }
      ctx.fillStyle = '#666'
      ctx.fillText('vocab size →', padL + plotW / 2, padT + plotH + 24)

      // coverage curve: sample geometrically
      ctx.strokeStyle = '#fbbf24'
      ctx.lineWidth = 2
      ctx.beginPath()
      const samples = 240
      for (let i = 0; i <= samples; i++) {
        const logV = (i / samples) * xMax
        const vSize = Math.max(1, Math.min(VOCAB_N, Math.round(Math.pow(10, logV))))
        const cv = coverageAt(cum, vSize)
        const sx = toSx(logV)
        const sy = toSy(cv)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()

      // Fill under curve softly
      ctx.globalAlpha = 0.08
      ctx.fillStyle = '#fbbf24'
      ctx.lineTo(toSx(xMax), toSy(0))
      ctx.lineTo(toSx(0), toSy(0))
      ctx.closePath()
      ctx.fill()
      ctx.globalAlpha = 1

      // Cutoff marker
      const sxC = toSx(vocabLog)
      const syC = toSy(cov)
      ctx.strokeStyle = 'rgba(74, 222, 128, 0.8)'
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(sxC, padT)
      ctx.lineTo(sxC, padT + plotH)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(padL, syC)
      ctx.lineTo(sxC, syC)
      ctx.stroke()
      ctx.setLineDash([])

      ctx.fillStyle = '#4ade80'
      ctx.beginPath()
      ctx.arc(sxC, syC, 4, 0, Math.PI * 2)
      ctx.fill()

      ctx.textAlign = 'left'
      ctx.fillStyle = '#4ade80'
      ctx.fillText(`|V| = ${vocab.toLocaleString()}  →  ${(cov * 100).toFixed(1)}% coverage`, sxC + 8, syC - 8)
    }

    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [vocabLog, cov, vocab, cum])

  const topTen = counts.slice(0, 10)
  const maxTop = topTen[0]

  return (
    <WidgetFrame
      widgetName="VocabularyStats"
      label="vocab size vs corpus coverage"
      right={<span className="font-mono">synthetic Zipfian · corpus = {CORPUS_SIZE.toLocaleString()} tokens</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="log₁₀|V|"
            value={vocabLog}
            min={0}
            max={Math.log10(VOCAB_N)}
            step={0.02}
            onChange={setVocabLog}
            format={(v) => `|V|=${Math.round(Math.pow(10, v)).toLocaleString()}`}
            accent="accent-term-amber"
          />
          <Slider
            label="Zipf s"
            value={s}
            min={0.8}
            max={1.4}
            step={0.01}
            onChange={setS}
            format={(v) => v.toFixed(2)}
            accent="accent-term-cyan"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="coverage" value={`${(cov * 100).toFixed(1)}%`} accent="text-term-green" />
            <Readout label="oov" value={`${((1 - cov) * 100).toFixed(1)}%`} accent="text-term-rose" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[1fr_240px] gap-4 overflow-hidden">
        {/* Coverage chart */}
        <div ref={boxRef} className="relative min-h-0">
          <canvas ref={canvasRef} className="w-full h-full block" />
        </div>

        {/* Top-10 bar chart */}
        <div className="flex flex-col gap-1.5 min-w-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            top-10 token frequencies
          </div>
          <div className="flex-1 flex flex-col gap-1 justify-center">
            {topTen.map((c, i) => {
              const pct = c / maxTop
              return (
                <div key={i} className="flex items-center gap-2 font-mono text-[10.5px]">
                  <span className="w-9 text-right text-dark-text-muted">{TOP_WORDS[i]}</span>
                  <div className="flex-1 h-4 bg-dark-surface-elevated/40 rounded-sm overflow-hidden">
                    <div
                      className="h-full"
                      style={{
                        width: `${pct * 100}%`,
                        background: `linear-gradient(90deg, rgba(251,191,36,0.8), rgba(251,191,36,0.35))`,
                      }}
                    />
                  </div>
                  <span className="w-12 text-right tabular-nums text-dark-text-primary">
                    {Math.round(c).toLocaleString()}
                  </span>
                </div>
              )
            })}
          </div>
          <div className="text-[9.5px] font-mono text-dark-text-disabled leading-snug mt-1">
            long-tail collapse: {(topTen.reduce((a, v) => a + v, 0) / CORPUS_SIZE * 100).toFixed(1)}% of all tokens come from just these 10.
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
