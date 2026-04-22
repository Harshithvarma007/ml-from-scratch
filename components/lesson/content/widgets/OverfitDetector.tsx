'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Simulated training. Dataset size slider drives the overfit gap. Toggles for
// dropout, weight decay, augmentation each reduce the gap. Train-loss curve
// stays roughly the same; val-loss curve responds to the regularisers.

function simulate(
  N: number,
  EPOCHS: number,
  dropout: boolean,
  wd: boolean,
  aug: boolean,
): { train: number[]; val: number[] } {
  const train: number[] = []
  const val: number[] = []

  // Overfit severity — small N → huge gap, large N → small gap.
  const baseGap = Math.max(0, 1.2 - Math.log10(N) * 0.5)
  // Each regulariser shrinks the gap
  let gapFactor = 1
  if (dropout) gapFactor *= 0.55
  if (wd) gapFactor *= 0.7
  if (aug) gapFactor *= 0.5

  const finalTrainLoss = 0.03 + (dropout || wd ? 0.05 : 0)
  const finalValLoss = finalTrainLoss + baseGap * gapFactor

  for (let e = 0; e < EPOCHS; e++) {
    const t = e / (EPOCHS - 1)
    const tl = 1.0 * Math.exp(-3 * t) + finalTrainLoss * (1 - Math.exp(-3 * t))
    train.push(tl)
    // Val: drops with train initially, then plateaus higher
    // If strongly overfitting, val curves back UP after midpoint
    const overfitSeverity = baseGap * gapFactor
    const valLossPath =
      1.0 * Math.exp(-3 * t) +
      finalValLoss * (1 - Math.exp(-3 * t)) +
      (overfitSeverity > 0.5 ? 0.3 * Math.pow(Math.max(0, t - 0.4), 2) : 0)
    val.push(valLossPath)
  }
  return { train, val }
}

const EPOCHS = 30

export default function OverfitDetector() {
  const [N, setN] = useState(200)
  const [dropout, setDropout] = useState(false)
  const [wd, setWd] = useState(false)
  const [aug, setAug] = useState(false)

  const { train, val } = useMemo(() => simulate(N, EPOCHS, dropout, wd, aug), [N, dropout, wd, aug])
  const gap = val[val.length - 1] - train[train.length - 1]

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
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, w, h)

      const padL = 48
      const padR = 12
      const padT = 18
      const padB = 28
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const yMax = 1.3
      const toSx = (e: number) => padL + (e / (EPOCHS - 1)) * plotW
      const toSy = (v: number) => padT + plotH - (Math.min(v, yMax) / yMax) * plotH

      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      ;[0, 0.3, 0.6, 0.9, 1.2].forEach((v) => {
        const sy = toSy(v)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(v.toFixed(1), padL - 6, sy + 3)
      })
      ctx.textAlign = 'center'
      ;[0, 10, 20, 29].forEach((e) => ctx.fillText(String(e), toSx(e), padT + plotH + 14))
      ctx.fillStyle = '#777'
      ctx.fillText('epoch', padL + plotW / 2, padT + plotH + 24)

      // Gap shading
      ctx.fillStyle = 'rgba(248, 113, 113, 0.08)'
      ctx.beginPath()
      train.forEach((v, i) => {
        const sx = toSx(i)
        const sy = toSy(v)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      for (let i = val.length - 1; i >= 0; i--) {
        ctx.lineTo(toSx(i), toSy(val[i]))
      }
      ctx.closePath()
      ctx.fill()

      // Train
      ctx.strokeStyle = '#4ade80'
      ctx.lineWidth = 2
      ctx.beginPath()
      train.forEach((v, i) => {
        const sx = toSx(i)
        const sy = toSy(v)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.stroke()

      // Val
      ctx.strokeStyle = '#f472b6'
      ctx.lineWidth = 2
      ctx.beginPath()
      val.forEach((v, i) => {
        const sx = toSx(i)
        const sy = toSy(v)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      })
      ctx.stroke()

      // Legend
      ctx.fillStyle = '#4ade80'
      ctx.fillRect(padL + 8, padT + 4, 10, 2)
      ctx.fillStyle = '#ccc'
      ctx.textAlign = 'left'
      ctx.fillText('train loss', padL + 22, padT + 8)
      ctx.fillStyle = '#f472b6'
      ctx.fillRect(padL + 100, padT + 4, 10, 2)
      ctx.fillStyle = '#ccc'
      ctx.fillText('val loss', padL + 114, padT + 8)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [train, val])

  const verdict =
    gap < 0.1 ? 'no overfit' : gap < 0.3 ? 'mild overfit' : gap < 0.7 ? 'overfit' : 'severe overfit'
  const verdictColor =
    gap < 0.1 ? 'text-term-green' : gap < 0.3 ? 'text-term-amber' : 'text-term-rose'

  return (
    <WidgetFrame
      widgetName="OverfitDetector"
      label="overfitting — the train/val gap and how to close it"
      right={<span className="font-mono">synthetic classifier · rose shading = generalisation gap</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="dataset N"
            value={N}
            min={20}
            max={5000}
            step={20}
            onChange={(v) => setN(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-purple"
          />
          <label className="flex items-center gap-2 cursor-pointer text-[11px] font-mono">
            <input type="checkbox" checked={dropout} onChange={(e) => setDropout(e.target.checked)} className="accent-term-green" />
            <span className="text-dark-text-secondary">dropout</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer text-[11px] font-mono">
            <input type="checkbox" checked={wd} onChange={(e) => setWd(e.target.checked)} className="accent-term-green" />
            <span className="text-dark-text-secondary">weight decay</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer text-[11px] font-mono">
            <input type="checkbox" checked={aug} onChange={(e) => setAug(e.target.checked)} className="accent-term-green" />
            <span className="text-dark-text-secondary">augmentation</span>
          </label>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="gap" value={gap.toFixed(3)} accent={verdictColor} />
            <Readout label="verdict" value={verdict} accent={verdictColor} />
          </div>
        </div>
      }
    >
      <div ref={boxRef} className="absolute inset-0">
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>
    </WidgetFrame>
  )
}
