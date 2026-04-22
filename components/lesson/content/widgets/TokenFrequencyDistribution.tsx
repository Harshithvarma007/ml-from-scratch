'use client'

// Classic Zipf log-log plot. Synthetic 5000-token corpus with frequency
// f(r) ∝ 1/r^s. The slider changes s, and the dashed reference line (pure
// 1/rank) is always shown for comparison. Top-20 tokens are labeled as
// highlighted points along the curve.

import { useMemo, useRef, useEffect, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'

const N = 5000
const TOP20_LABELS = [
  'the', 'of', 'and', 'to', 'a', 'in', 'is', 'it', 'that', 'you',
  'he', 'for', 'was', 'on', 'are', 'with', 'as', 'at', 'by', 'this',
]

function buildFreq(s: number): number[] {
  // f(r) = C / r^s, normalized so sum = 1
  const raw: number[] = []
  let sum = 0
  for (let r = 1; r <= N; r++) {
    const v = 1 / Math.pow(r, s)
    raw.push(v)
    sum += v
  }
  return raw.map((v) => v / sum)
}

export default function TokenFrequencyDistribution() {
  const [s, setS] = useState(1.0)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)
  const [hoverRank, setHoverRank] = useState<number | null>(null)

  const freq = useMemo(() => buildFreq(s), [s])

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

      const padL = 58
      const padR = 14
      const padT = 14
      const padB = 34
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const logRMin = 0                              // log(1)
      const logRMax = Math.log10(N)
      const logFMax = Math.log10(freq[0])
      const logFMin = Math.log10(freq[N - 1])
      const toSx = (r: number) => padL + ((Math.log10(r) - logRMin) / (logRMax - logRMin)) * plotW
      const toSy = (f: number) => padT + plotH - ((Math.log10(f) - logFMin) / (logFMax - logFMin)) * plotH

      // grid
      ctx.font = '9.5px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.strokeStyle = '#1e1e1e'
      for (const r of [1, 10, 100, 1000, 5000]) {
        const sx = toSx(r)
        ctx.beginPath()
        ctx.moveTo(sx, padT)
        ctx.lineTo(sx, padT + plotH)
        ctx.stroke()
        ctx.textAlign = 'center'
        ctx.fillText(String(r), sx, padT + plotH + 14)
      }
      // y-axis decades
      const lo = Math.ceil(logFMin)
      const hi = Math.floor(logFMax)
      for (let e = lo; e <= hi; e++) {
        const f = Math.pow(10, e)
        const sy = toSy(f)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.textAlign = 'right'
        ctx.fillText(`1e${e}`, padL - 6, sy + 3)
      }
      ctx.fillStyle = '#777'
      ctx.textAlign = 'center'
      ctx.fillText('rank (log)', padL + plotW / 2, padT + plotH + 28)
      ctx.save()
      ctx.translate(14, padT + plotH / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.fillText('relative frequency (log)', 0, 0)
      ctx.restore()

      // reference line: pure 1/rank (s = 1), normalized same way
      const refFreq = (r: number) => 1 / (r * (Math.log(N) + 0.5772 + 0.5 / N))
      ctx.strokeStyle = 'rgba(167, 139, 250, 0.55)'
      ctx.setLineDash([4, 4])
      ctx.lineWidth = 1.4
      ctx.beginPath()
      for (let i = 0; i < 200; i++) {
        const r = Math.pow(10, (i / 199) * logRMax)
        const fv = refFreq(r)
        const sx = toSx(r)
        const sy = toSy(fv)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()
      ctx.setLineDash([])

      // actual curve
      ctx.strokeStyle = '#67e8f9'
      ctx.lineWidth = 2
      ctx.beginPath()
      for (let r = 1; r <= N; r++) {
        const sx = toSx(r)
        const sy = toSy(freq[r - 1])
        if (r === 1) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()

      // top-20 labeled points
      for (let i = 0; i < 20; i++) {
        const r = i + 1
        const f = freq[r - 1]
        const sx = toSx(r)
        const sy = toSy(f)
        ctx.fillStyle = '#fbbf24'
        ctx.beginPath()
        ctx.arc(sx, sy, 3, 0, Math.PI * 2)
        ctx.fill()
        if (i < 10) {
          ctx.fillStyle = '#fbbf24'
          ctx.font = '9px "JetBrains Mono", monospace'
          ctx.textAlign = 'left'
          ctx.fillText(`"${TOP20_LABELS[i]}"`, sx + 5, sy - 5)
        }
      }

      // hover cursor
      if (hoverRank !== null) {
        const sx = toSx(hoverRank)
        const sy = toSy(freq[hoverRank - 1])
        ctx.strokeStyle = 'rgba(255,255,255,0.3)'
        ctx.setLineDash([2, 3])
        ctx.beginPath()
        ctx.moveTo(sx, padT)
        ctx.lineTo(sx, padT + plotH)
        ctx.stroke()
        ctx.setLineDash([])
        ctx.fillStyle = '#f472b6'
        ctx.beginPath()
        ctx.arc(sx, sy, 4, 0, Math.PI * 2)
        ctx.fill()
      }

      // legend
      const ly = padT + 12
      ctx.fillStyle = '#ccc'
      ctx.textAlign = 'left'
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.strokeStyle = '#67e8f9'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(padL + plotW - 180, ly - 4)
      ctx.lineTo(padL + plotW - 165, ly - 4)
      ctx.stroke()
      ctx.fillText(`1/r^${s.toFixed(2)}`, padL + plotW - 160, ly)
      ctx.strokeStyle = '#a78bfa'
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(padL + plotW - 90, ly - 4)
      ctx.lineTo(padL + plotW - 75, ly - 4)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillText('pure 1/r', padL + plotW - 70, ly)
    }

    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [freq, s, hoverRank])

  const handleMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const box = boxRef.current
    if (!box) return
    const r = box.getBoundingClientRect()
    const x = e.clientX - r.left
    const padL = 58
    const plotW = r.width - padL - 14
    const logR = ((x - padL) / plotW) * Math.log10(N)
    const rank = Math.max(1, Math.min(N, Math.round(Math.pow(10, logR))))
    setHoverRank(rank)
  }

  const hoverF = hoverRank !== null ? freq[hoverRank - 1] : 0

  return (
    <WidgetFrame
      widgetName="TokenFrequencyDistribution"
      label="token frequency — log-log Zipf curve"
      right={<span className="font-mono">N = {N.toLocaleString()} unique tokens</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-3 flex-1 min-w-[260px] font-mono text-[12px]">
            <span className="text-dark-text-secondary whitespace-nowrap">Zipf exponent s</span>
            <input
              type="range"
              min={0.7}
              max={1.3}
              step={0.01}
              value={s}
              onChange={(e) => setS(Number(e.target.value))}
              className="flex-1 min-w-0 h-1 rounded-full bg-dark-border cursor-pointer accent-term-cyan"
            />
            <span className="text-dark-text-primary tabular-nums w-14 text-right">{s.toFixed(2)}</span>
          </label>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="rank" value={hoverRank !== null ? String(hoverRank) : '—'} accent="text-term-pink" />
            <Readout label="freq" value={hoverRank !== null ? hoverF.toExponential(2) : '—'} accent="text-term-cyan" />
            <Readout
              label="top-20 share"
              value={`${(freq.slice(0, 20).reduce((a, b) => a + b, 0) * 100).toFixed(1)}%`}
              accent="text-term-amber"
            />
          </div>
        </div>
      }
    >
      <div
        ref={boxRef}
        className="absolute inset-0"
        onMouseMove={handleMove}
        onMouseLeave={() => setHoverRank(null)}
      >
        <canvas ref={canvasRef} className="w-full h-full block cursor-crosshair" />
      </div>
    </WidgetFrame>
  )
}
