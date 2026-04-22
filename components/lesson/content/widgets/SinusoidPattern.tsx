'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Sinusoidal positional encoding heatmap. Rows = positions, cols = dims. Even
// dims use sin(pos/10000^(2i/d)); odd dims use cos(...). Click a row to pull
// up its raw values as a line chart. A red-to-blue diverging colormap makes
// the characteristic frequency bands visible.

function peValue(pos: number, i: number, dModel: number): number {
  const freq = 1 / Math.pow(10000, (2 * Math.floor(i / 2)) / dModel)
  return i % 2 === 0 ? Math.sin(pos * freq) : Math.cos(pos * freq)
}

function colorMap(v: number): string {
  // -1 → deep blue · 0 → near-black · +1 → deep red
  const x = Math.max(-1, Math.min(1, v))
  if (x >= 0) {
    const a = x
    const r = Math.round(40 + a * 215)
    const g = Math.round(30 + a * 40)
    const b = Math.round(30 + a * 60)
    return `rgb(${r}, ${g}, ${b})`
  } else {
    const a = -x
    const r = Math.round(30 + a * 40)
    const g = Math.round(30 + a * 90)
    const b = Math.round(50 + a * 200)
    return `rgb(${r}, ${g}, ${b})`
  }
}

export default function SinusoidPattern() {
  const [maxPos, setMaxPos] = useState(64)
  const [dModel, setDModel] = useState(128)
  const [selRow, setSelRow] = useState(0)

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)
  const lineRef = useRef<HTMLCanvasElement | null>(null)
  const lineBoxRef = useRef<HTMLDivElement | null>(null)

  const rowValues = useMemo(() => {
    const out = new Array(dModel)
    for (let i = 0; i < dModel; i++) out[i] = peValue(selRow, i, dModel)
    return out
  }, [selRow, dModel])

  // Draw heatmap
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

      const padL = 36
      const padT = 16
      const padB = 22
      const padR = 12
      const gridW = w - padL - padR
      const gridH = h - padT - padB
      const cellW = gridW / dModel
      const cellH = gridH / maxPos

      for (let pos = 0; pos < maxPos; pos++) {
        for (let i = 0; i < dModel; i++) {
          const v = peValue(pos, i, dModel)
          ctx.fillStyle = colorMap(v)
          ctx.fillRect(
            padL + i * cellW,
            padT + pos * cellH,
            Math.max(1, cellW + 0.5),
            Math.max(1, cellH + 0.5),
          )
        }
      }

      // Axis labels
      ctx.font = '9.5px "JetBrains Mono", monospace'
      ctx.fillStyle = '#666'
      ctx.textAlign = 'right'
      for (let p = 0; p <= maxPos; p += Math.max(1, Math.floor(maxPos / 8))) {
        if (p > maxPos) break
        ctx.fillText(String(p), padL - 4, padT + p * cellH + 3)
      }
      ctx.textAlign = 'center'
      for (let i = 0; i <= dModel; i += Math.max(1, Math.floor(dModel / 8))) {
        if (i > dModel) break
        ctx.fillText(String(i), padL + i * cellW, padT + gridH + 14)
      }
      ctx.fillStyle = '#888'
      ctx.textAlign = 'center'
      ctx.fillText('dim →', padL + gridW / 2, padT + gridH + 14 + 8)
      ctx.save()
      ctx.translate(padL - 24, padT + gridH / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.fillText('pos →', 0, 0)
      ctx.restore()

      // Selected row highlight
      if (selRow < maxPos) {
        ctx.strokeStyle = '#fbbf24'
        ctx.lineWidth = 1.5
        ctx.strokeRect(padL - 1, padT + selRow * cellH - 1, gridW + 2, cellH + 2)
      }
    }

    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [maxPos, dModel, selRow])

  // Click handler to pick a row
  const onHeatmapClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const box = boxRef.current
    if (!box) return
    const r = box.getBoundingClientRect()
    const y = e.clientY - r.top
    const padT = 16
    const padB = 22
    const gridH = r.height - padT - padB
    const cellH = gridH / maxPos
    const idx = Math.floor((y - padT) / cellH)
    if (idx >= 0 && idx < maxPos) setSelRow(idx)
  }

  // Draw line chart for selected row
  useEffect(() => {
    const canvas = lineRef.current
    const box = lineBoxRef.current
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

      const padL = 28
      const padR = 8
      const padT = 8
      const padB = 16
      const plotW = w - padL - padR
      const plotH = h - padT - padB
      const toSx = (i: number) => padL + (i / (dModel - 1)) * plotW
      const toSy = (v: number) => padT + plotH - ((v + 1) / 2) * plotH

      // Zero line
      ctx.strokeStyle = '#2a2a32'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(padL, toSy(0))
      ctx.lineTo(padL + plotW, toSy(0))
      ctx.stroke()

      // Trace
      ctx.strokeStyle = '#fbbf24'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      for (let i = 0; i < dModel; i++) {
        const sx = toSx(i)
        const sy = toSy(rowValues[i])
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()

      ctx.font = '9px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      ctx.fillText('+1', padL - 4, toSy(1) + 3)
      ctx.fillText(' 0', padL - 4, toSy(0) + 3)
      ctx.fillText('−1', padL - 4, toSy(-1) + 3)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [rowValues, dModel])

  // Period at the user-selected row's first dim for a readout
  const dimIdx = Math.min(selRow * 2, dModel - 2)
  const freq = 1 / Math.pow(10000, (2 * Math.floor(dimIdx / 2)) / dModel)
  const period = (2 * Math.PI) / freq

  return (
    <WidgetFrame
      widgetName="SinusoidPattern"
      label="sinusoidal positional encoding — red = +1, blue = −1"
      right={<span className="font-mono">PE[pos, 2i] = sin(pos / 10000^(2i/d)) · PE[pos, 2i+1] = cos(...)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="max_pos"
            value={maxPos}
            min={16}
            max={128}
            step={1}
            onChange={(v) => { setMaxPos(Math.round(v)); if (selRow >= v) setSelRow(0) }}
            format={(v) => String(Math.round(v))}
            accent="accent-term-amber"
          />
          <Slider
            label="d_model"
            value={dModel}
            min={16}
            max={256}
            step={1}
            onChange={(v) => setDModel(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-cyan"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="row" value={String(selRow)} accent="text-term-amber" />
            <Readout label="period at dim 2" value={`≈ ${period.toFixed(1)}`} accent="text-term-cyan" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-rows-[1fr_110px] gap-3 overflow-hidden">
        <div ref={boxRef} className="relative min-h-0 cursor-crosshair" onClick={onHeatmapClick}>
          <canvas ref={canvasRef} className="w-full h-full block" />
        </div>
        <div className="flex flex-col gap-1 min-h-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            row {selRow}: PE[{selRow}, :] — click the heatmap to pick a row
          </div>
          <div ref={lineBoxRef} className="relative flex-1 bg-dark-surface-elevated/20 rounded border border-dark-border">
            <canvas ref={lineRef} className="w-full h-full block" />
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
