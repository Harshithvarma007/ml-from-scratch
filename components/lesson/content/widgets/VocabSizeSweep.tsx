'use client'

// Vocab size vs tokens-per-document curve. Uses a synthetic Zipfian corpus so
// the curve has the right shape: sharp drop, then diminishing returns. The
// slider picks a vocab size; we estimate tokens-per-document and draw a 10×10
// preview grid of what the vocab contains at that point. A "sweet spot" band
// is drawn near 32k tokens where most production LLMs live.

import { useMemo, useRef, useEffect, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Synthetic Zipfian corpus: rank-R token has frequency proportional to 1/R^s.
// Total unique words in the "true" word list = 50k. Tokens-per-document for
// a given vocab size V is approximated by:
//   - for the top V entries, you emit 1 token each
//   - for every word outside V, you emit an average of `subwords_per_oov`
//     tokens (grows as V shrinks, because the BPE has to fall back further)
const CORPUS_UNIQUE = 50000
const ZIPF_S = 1.0
const DOC_LEN_WORDS = 800 // typical doc length in words
const MIN_LOG_V = Math.log(256)
const MAX_LOG_V = Math.log(100000)

function harmonic(n: number, s: number): number {
  // H_n^(s) with small-n correction; we only need approximate values
  let sum = 0
  for (let k = 1; k <= n; k++) sum += 1 / Math.pow(k, s)
  return sum
}

const H_TOTAL = harmonic(CORPUS_UNIQUE, ZIPF_S)

// Expected tokens-per-document as vocab size V grows.
// Fraction-of-words-covered-by-single-token = H_V / H_total.
// For uncovered words, subword count scales as log(word_len) / log(V_scale) —
// we parameterize simply as 1.2 + 4 * (1 - V/CORPUS_UNIQUE)^2.
function tokensPerDoc(V: number): number {
  const Vc = Math.min(V, CORPUS_UNIQUE)
  const covered = harmonic(Vc, ZIPF_S) / H_TOTAL
  const uncovered = 1 - covered
  const subwords = 1.2 + 4 * Math.pow(1 - Vc / CORPUS_UNIQUE, 2)
  return DOC_LEN_WORDS * (covered * 1.0 + uncovered * subwords)
}

// Sample "token" labels in rank order. This is visual only; we don't care
// about real English. We use a few buckets to color-code.
function fakeTokenAtRank(r: number): { text: string; kind: 'top' | 'mid' | 'tail' } {
  // top 200 are common English words
  const COMMON = [
    'the', '▁of', '▁and', '▁a', '▁to', '▁in', '▁is', '▁it', '▁that', '▁he',
    '▁for', '▁was', '▁on', '▁are', '▁as', '▁with', '▁his', '▁they', '▁at', '▁be',
    '▁this', '▁have', '▁from', '▁or', '▁had', '▁by', '▁not', '▁but', '▁what', '▁all',
    '▁were', '▁we', '▁when', '▁your', '▁can', '▁said', '▁there', '▁use', '▁an', '▁each',
    'ing', 'ed', 'er', 'ly', 'tion', 'ness', 'ment', 'able', '▁the', '▁and',
  ]
  if (r < COMMON.length) return { text: COMMON[r], kind: 'top' }
  if (r < 5000) {
    const stems = ['▁run', '▁hand', '▁walk', '▁work', '▁play', '▁read', '▁open', '▁turn', '▁make', '▁stop']
    const suff = ['ing', 'ed', 's', 'er', 'ly']
    return { text: stems[r % stems.length] + suff[(r >> 3) % suff.length], kind: 'mid' }
  }
  // tail: suffix-y / morpheme shards
  const shards = ['ph', 'qu', 'tz', 'xy', 'zz', 'kk', 'wr', 'yl', 'sk', 'gh', 'ch', 'sh']
  return { text: shards[r % shards.length] + shards[(r * 7) % shards.length], kind: 'tail' }
}

export default function VocabSizeSweep() {
  const [logV, setLogV] = useState(Math.log(32000))
  const V = Math.round(Math.exp(logV))
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const { curve, sampled } = useMemo(() => {
    const N = 64
    const curve: [number, number][] = []
    for (let i = 0; i < N; i++) {
      const lv = MIN_LOG_V + (i / (N - 1)) * (MAX_LOG_V - MIN_LOG_V)
      const vv = Math.exp(lv)
      curve.push([vv, tokensPerDoc(vv)])
    }
    return { curve, sampled: tokensPerDoc(V) }
  }, [V])

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

      const padL = 48
      const padR = 14
      const padT = 14
      const padB = 28
      const plotW = w - padL - padR
      const plotH = h - padT - padB
      const yMax = curve[0][1] * 1.05
      const yMin = curve[curve.length - 1][1] * 0.7
      const toSx = (v: number) => padL + ((Math.log(v) - MIN_LOG_V) / (MAX_LOG_V - MIN_LOG_V)) * plotW
      const toSy = (t: number) => padT + plotH - ((t - yMin) / (yMax - yMin)) * plotH

      // grid
      ctx.font = '9.5px "JetBrains Mono", monospace'
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.fillStyle = '#555'
      for (const v of [256, 1024, 4096, 16384, 32000, 65536, 100000]) {
        if (v < Math.exp(MIN_LOG_V) || v > Math.exp(MAX_LOG_V)) continue
        const sx = toSx(v)
        ctx.beginPath()
        ctx.moveTo(sx, padT)
        ctx.lineTo(sx, padT + plotH)
        ctx.stroke()
        ctx.textAlign = 'center'
        ctx.fillText(v >= 1000 ? `${Math.round(v / 1000)}k` : String(v), sx, padT + plotH + 14)
      }
      for (let k = 0; k < 4; k++) {
        const y = yMin + (k / 3) * (yMax - yMin)
        const sy = toSy(y)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.textAlign = 'right'
        ctx.fillText(Math.round(y).toString(), padL - 6, sy + 3)
      }
      ctx.fillStyle = '#777'
      ctx.textAlign = 'center'
      ctx.fillText('vocab size (log scale)', padL + plotW / 2, padT + plotH + 26)
      ctx.save()
      ctx.translate(14, padT + plotH / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.fillText('tokens per doc (800 words)', 0, 0)
      ctx.restore()

      // sweet spot band: 16k - 50k
      const x1 = toSx(16000)
      const x2 = toSx(50000)
      ctx.fillStyle = 'rgba(74, 222, 128, 0.07)'
      ctx.fillRect(x1, padT, x2 - x1, plotH)
      ctx.strokeStyle = 'rgba(74, 222, 128, 0.25)'
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(x1, padT)
      ctx.lineTo(x1, padT + plotH)
      ctx.moveTo(x2, padT)
      ctx.lineTo(x2, padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = '#4ade80'
      ctx.textAlign = 'center'
      ctx.fillText('sweet spot', (x1 + x2) / 2, padT + 12)

      // curve
      ctx.strokeStyle = '#67e8f9'
      ctx.lineWidth = 2
      ctx.beginPath()
      curve.forEach(([v, t], i) => {
        const sx = toSx(v)
        const sy = toSy(t)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.stroke()

      // selected point
      const sx = toSx(V)
      const sy = toSy(sampled)
      ctx.strokeStyle = 'rgba(251, 191, 36, 0.5)'
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(sx, padT)
      ctx.lineTo(sx, padT + plotH)
      ctx.moveTo(padL, sy)
      ctx.lineTo(padL + plotW, sy)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = '#fbbf24'
      ctx.beginPath()
      ctx.arc(sx, sy, 5, 0, Math.PI * 2)
      ctx.fill()
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [curve, sampled, V])

  // Build the 10×10 preview grid
  const gridTokens = useMemo(() => {
    const N = 100
    const out: { rank: number; token: string; kind: string }[] = []
    for (let i = 0; i < N; i++) {
      // sample ranks spanning from 0 to V-1 roughly uniformly in log
      const frac = i / (N - 1)
      const r = Math.min(V - 1, Math.floor(Math.pow(V, Math.max(0.1, frac))))
      const { text, kind } = fakeTokenAtRank(r)
      out.push({ rank: r, token: text, kind })
    }
    return out
  }, [V])

  const inSweet = V >= 16000 && V <= 50000

  return (
    <WidgetFrame
      widgetName="VocabSizeSweep"
      label="vocab size vs tokens per document · Zipfian synthetic corpus"
      right={<span className="font-mono">|corpus| = 50k unique words · s = 1.0</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-3 flex-1 min-w-[260px] font-mono text-[12px]">
            <span className="text-dark-text-secondary whitespace-nowrap">vocab V</span>
            <input
              type="range"
              min={MIN_LOG_V}
              max={MAX_LOG_V}
              step={0.01}
              value={logV}
              onChange={(e) => setLogV(Number(e.target.value))}
              className="flex-1 min-w-0 h-1 rounded-full bg-dark-border cursor-pointer accent-term-amber"
            />
            <span className="text-dark-text-primary tabular-nums w-20 text-right">
              {V >= 1000 ? `${(V / 1000).toFixed(1)}k` : V}
            </span>
          </label>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="tok/doc" value={Math.round(sampled).toLocaleString()} accent="text-term-cyan" />
            <Readout
              label="zone"
              value={inSweet ? 'sweet spot' : V < 16000 ? 'too small' : 'too big'}
              accent={inSweet ? 'text-term-green' : 'text-term-rose'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[1fr_220px] gap-4 overflow-hidden">
        <div ref={boxRef} className="relative min-h-0">
          <canvas ref={canvasRef} className="w-full h-full block" />
        </div>
        <div className="flex flex-col gap-2 min-h-0 overflow-hidden">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            vocab preview · 100-token sample
          </div>
          <div className="grid grid-cols-10 gap-[2px] flex-1 min-h-0 overflow-hidden">
            {gridTokens.map((g, i) => (
              <div
                key={i}
                title={`rank ${g.rank} · "${g.token}"`}
                className={cn(
                  'h-auto aspect-square rounded-[2px] text-[7px] font-mono flex items-center justify-center truncate px-[1px]',
                  g.kind === 'top' && 'bg-term-amber/20 text-term-amber',
                  g.kind === 'mid' && 'bg-term-cyan/15 text-term-cyan',
                  g.kind === 'tail' && 'bg-term-rose/10 text-term-rose/80',
                )}
              >
                {g.token.replace(/▁/g, '\u00b7').slice(0, 4)}
              </div>
            ))}
          </div>
          <div className="text-[9.5px] font-mono text-dark-text-disabled leading-snug">
            amber = top common words · cyan = morphemes · rose = long-tail shards
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
