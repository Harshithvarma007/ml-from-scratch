'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Plot speculation length k vs. expected speedup. With per-token acceptance
// probability α, the expected number of accepted draft tokens in a k-step
// window is:
//   E[accepts] = α · (1 − α^k) / (1 − α)     (sum of geometric series, clipped at k)
// Counting the bonus token emitted when all k are accepted (equivalent to
// the "+1" in the standard spec-decoding analysis):
//   E[commits] = E[accepts] + (α^k) ·  ... ≈ (1 − α^(k+1)) / (1 − α)
// One full round costs ~ (1 + k·c) target-equivalents where c ≈ draft/target
// cost. Expected speedup = E[commits] / (1 + k·c).
// We plot speedup vs. k for a user-tunable α and draft cost, then mark the
// optimal k.

function expectedAccepts(alpha: number, k: number): number {
  if (alpha >= 1) return k + 1
  if (alpha === 0) return 1
  return (1 - Math.pow(alpha, k + 1)) / (1 - alpha)
}

function speedup(alpha: number, k: number, draftCost: number): number {
  const commits = expectedAccepts(alpha, k)
  const cost = 1 + k * draftCost
  return commits / cost
}

export default function AcceptanceRate() {
  const [alpha, setAlpha] = useState(0.75)
  const [draftCost, setDraftCost] = useState(0.15)

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const { bestK, bestSpeedup, series } = useMemo(() => {
    const s: { k: number; sp: number }[] = []
    for (let k = 1; k <= 16; k++) {
      s.push({ k, sp: speedup(alpha, k, draftCost) })
    }
    let bestK = 1
    let bestSpeedup = 0
    for (const p of s) {
      if (p.sp > bestSpeedup) {
        bestSpeedup = p.sp
        bestK = p.k
      }
    }
    return { bestK, bestSpeedup, series: s }
  }, [alpha, draftCost])

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
      ctx.setTransform(1, 0, 0, 1, 0, 0)
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, w, h)

      const padL = 46
      const padR = 18
      const padT = 18
      const padB = 34
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const xMin = 1
      const xMax = 16
      const yMax = Math.max(bestSpeedup * 1.15, 2.5)
      const yMin = 0.5
      const toSx = (k: number) => padL + ((k - xMin) / (xMax - xMin)) * plotW
      const toSy = (v: number) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH

      // grid
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      for (let v = 1; v <= Math.ceil(yMax); v += 0.5) {
        if (v < yMin) continue
        ctx.beginPath()
        ctx.moveTo(padL, toSy(v))
        ctx.lineTo(padL + plotW, toSy(v))
        ctx.stroke()
        ctx.textAlign = 'right'
        ctx.fillText(v.toFixed(1), padL - 6, toSy(v) + 3)
      }
      for (let k = 1; k <= 16; k += 3) {
        ctx.textAlign = 'center'
        ctx.fillText(String(k), toSx(k), padT + plotH + 14)
      }
      ctx.fillText('k (speculation length)', padL + plotW / 2, padT + plotH + 28)
      ctx.save()
      ctx.translate(14, padT + plotH / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.textAlign = 'center'
      ctx.fillText('expected speedup', 0, 0)
      ctx.restore()

      // baseline at speedup = 1
      ctx.strokeStyle = '#3f3f46'
      ctx.lineWidth = 1
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(padL, toSy(1))
      ctx.lineTo(padL + plotW, toSy(1))
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = '#777'
      ctx.textAlign = 'left'
      ctx.fillText('baseline (no speculation)', padL + 6, toSy(1) - 4)

      // series line (amber)
      ctx.strokeStyle = '#fbbf24'
      ctx.lineWidth = 2
      ctx.beginPath()
      series.forEach((p, i) => {
        const x = toSx(p.k)
        const y = toSy(p.sp)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()

      // dots
      series.forEach((p) => {
        ctx.fillStyle = p.k === bestK ? '#4ade80' : '#fbbf24'
        ctx.beginPath()
        ctx.arc(toSx(p.k), toSy(p.sp), p.k === bestK ? 5 : 3, 0, Math.PI * 2)
        ctx.fill()
      })

      // best marker
      ctx.strokeStyle = '#4ade80'
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(toSx(bestK), padT)
      ctx.lineTo(toSx(bestK), padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = '#4ade80'
      ctx.textAlign = 'left'
      ctx.fillText(`k* = ${bestK}`, toSx(bestK) + 8, padT + 14)
      ctx.fillText(`${bestSpeedup.toFixed(2)}×`, toSx(bestK) + 8, padT + 28)

      // extra curves for α ± 0.1 as ghosts
      ;[-0.1, 0.1].forEach((da, j) => {
        const a2 = Math.max(0.01, Math.min(0.99, alpha + da))
        ctx.strokeStyle = j === 0 ? 'rgba(103, 232, 249, 0.35)' : 'rgba(247, 113, 133, 0.35)'
        ctx.lineWidth = 1.2
        ctx.setLineDash([2, 3])
        ctx.beginPath()
        for (let k = 1; k <= 16; k++) {
          const sp = speedup(a2, k, draftCost)
          const x = toSx(k)
          const y = toSy(sp)
          if (k === 1) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.stroke()
        ctx.setLineDash([])
        ctx.fillStyle = j === 0 ? '#67e8f9' : '#fb7185'
        ctx.textAlign = 'left'
        ctx.fillText(`α = ${a2.toFixed(2)}`, padL + plotW - 72, j === 0 ? padT + 14 : padT + 28)
      })

      ctx.fillStyle = '#fbbf24'
      ctx.textAlign = 'left'
      ctx.fillText(`α = ${alpha.toFixed(2)}`, padL + plotW - 72, padT + 42)
    }

    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [alpha, draftCost, bestK, bestSpeedup, series])

  const acceptedPerRound = expectedAccepts(alpha, bestK)

  return (
    <WidgetFrame
      widgetName="AcceptanceRate"
      label="acceptance α vs. optimal k — the speedup curve"
      right={<span className="font-mono">speedup = (1 − α^(k+1)) / [(1 − α) · (1 + k·c)]</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="α  (accept prob)"
            value={alpha}
            min={0.3}
            max={0.95}
            step={0.01}
            onChange={setAlpha}
            format={(v) => v.toFixed(2)}
            accent="accent-term-amber"
          />
          <Slider
            label="draft cost c"
            value={draftCost}
            min={0.03}
            max={0.5}
            step={0.01}
            onChange={setDraftCost}
            format={(v) => v.toFixed(2)}
            accent="accent-term-cyan"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="optimal k" value={String(bestK)} accent="text-term-green" />
            <Readout label="speedup" value={`${bestSpeedup.toFixed(2)}×`} accent="text-term-green" />
            <Readout label="E[accepts]" value={acceptedPerRound.toFixed(2)} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div ref={boxRef} className="absolute inset-0 p-4">
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>
    </WidgetFrame>
  )
}
