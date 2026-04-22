'use client'

import { useEffect, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Learning-rate schedules plotted against training step. The user can pick
// among five schedules and drag a cursor along the x-axis to read the
// instantaneous LR. Sliders: peak_lr, warmup_steps, total_steps. All five
// curves are drawn faintly; the selected one is highlighted.

type Schedule = 'warmup' | 'cosine' | 'linear' | 'wu_cosine' | 'constant'

const SCHEDULES: { key: Schedule; name: string; color: string; desc: string }[] = [
  {
    key: 'warmup',
    name: 'warmup only',
    color: '#67e8f9',
    desc: 'linear ramp from 0 → peak over warmup_steps, then hold peak forever. rarely used at scale.',
  },
  {
    key: 'cosine',
    name: 'cosine decay',
    color: '#a78bfa',
    desc: 'start at peak, decay half-cosine to 0 (or a floor) over total_steps. no warmup.',
  },
  {
    key: 'linear',
    name: 'linear decay',
    color: '#f472b6',
    desc: 'start at peak, linearly decay to 0 over total_steps. simplest schedule.',
  },
  {
    key: 'wu_cosine',
    name: 'warmup + cosine',
    color: '#fbbf24',
    desc: 'default GPT recipe: linear warmup to peak, then half-cosine to 10% of peak. the one you want by default.',
  },
  {
    key: 'constant',
    name: 'constant',
    color: '#f87171',
    desc: 'just peak_lr, always. instable at the start for any real model.',
  },
]

function lrAt(kind: Schedule, step: number, peak: number, warmup: number, total: number): number {
  const t = Math.min(step, total)
  const floor = 0.1 * peak
  switch (kind) {
    case 'warmup':
      return t < warmup ? peak * (t / warmup) : peak
    case 'cosine': {
      const p = t / total
      return 0.5 * (peak - floor) * (1 + Math.cos(Math.PI * p)) + floor
    }
    case 'linear':
      return peak * (1 - t / total)
    case 'wu_cosine': {
      if (t < warmup) return peak * (t / warmup)
      const p = (t - warmup) / Math.max(1, total - warmup)
      return 0.5 * (peak - floor) * (1 + Math.cos(Math.PI * p)) + floor
    }
    case 'constant':
      return peak
  }
}

export default function LRScheduleViz() {
  const [kind, setKind] = useState<Schedule>('wu_cosine')
  const [peak, setPeak] = useState(6e-4)
  const [warmup, setWarmup] = useState(2000)
  const [total, setTotal] = useState(20000)
  const [cursor, setCursor] = useState(5000)

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

      const padL = 58
      const padR = 14
      const padT = 14
      const padB = 28
      const plotW = w - padL - padR
      const plotH = h - padT - padB
      const yMax = peak * 1.08
      const yMin = 0
      const toSx = (s: number) => padL + (s / total) * plotW
      const toSy = (v: number) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH

      // Gridlines
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      for (let k = 0; k <= 4; k++) {
        const v = (yMax * k) / 4
        const sy = toSy(v)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(v.toExponential(1), padL - 6, sy + 3)
      }
      ctx.textAlign = 'center'
      for (let k = 0; k <= 4; k++) {
        const s = Math.round((total * k) / 4)
        ctx.fillText(String(s), toSx(s), padT + plotH + 14)
      }
      ctx.fillStyle = '#777'
      ctx.fillText('training step', padL + plotW / 2, padT + plotH + 24)

      // warmup region
      if (kind === 'warmup' || kind === 'wu_cosine') {
        ctx.fillStyle = 'rgba(103, 232, 249, 0.08)'
        ctx.fillRect(padL, padT, Math.min(warmup, total) / total * plotW, plotH)
      }

      // Faint background traces for non-selected schedules
      SCHEDULES.forEach((s) => {
        if (s.key === kind) return
        ctx.strokeStyle = s.color
        ctx.globalAlpha = 0.18
        ctx.lineWidth = 1.2
        ctx.beginPath()
        for (let i = 0; i <= 200; i++) {
          const st = (i / 200) * total
          const v = lrAt(s.key, st, peak, warmup, total)
          const sx = toSx(st)
          const sy = toSy(v)
          if (i === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        ctx.stroke()
      })
      ctx.globalAlpha = 1

      // Selected schedule curve
      const sel = SCHEDULES.find((s) => s.key === kind)!
      ctx.strokeStyle = sel.color
      ctx.lineWidth = 2.2
      ctx.beginPath()
      for (let i = 0; i <= 400; i++) {
        const st = (i / 400) * total
        const v = lrAt(kind, st, peak, warmup, total)
        const sx = toSx(st)
        const sy = toSy(v)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()

      // Cursor
      const sxC = toSx(cursor)
      ctx.strokeStyle = 'rgba(255,255,255,0.4)'
      ctx.setLineDash([3, 4])
      ctx.beginPath()
      ctx.moveTo(sxC, padT)
      ctx.lineTo(sxC, padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])
      const curLR = lrAt(kind, cursor, peak, warmup, total)
      ctx.fillStyle = sel.color
      ctx.beginPath()
      ctx.arc(sxC, toSy(curLR), 4.5, 0, Math.PI * 2)
      ctx.fill()

      // legend
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.textAlign = 'left'
      ctx.fillStyle = sel.color
      ctx.fillText(sel.name, padL + 10, padT + 14)
      ctx.fillStyle = '#777'
      ctx.fillText(`LR(step=${cursor}) = ${curLR.toExponential(2)}`, padL + 10, padT + 28)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [kind, peak, warmup, total, cursor])

  const sel = SCHEDULES.find((s) => s.key === kind)!
  const curLR = lrAt(kind, cursor, peak, warmup, total)

  return (
    <WidgetFrame
      widgetName="LRScheduleViz"
      label="learning-rate schedules — drag cursor to sample LR at a step"
      right={<span className="font-mono">warmup = {warmup} · total = {total}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="peak_lr"
            value={peak * 1e5}
            min={1}
            max={100}
            step={1}
            onChange={(v) => setPeak(v / 1e5)}
            format={() => peak.toExponential(1)}
            accent="accent-term-amber"
          />
          <Slider
            label="warmup"
            value={warmup}
            min={0}
            max={10000}
            step={100}
            onChange={(v) => setWarmup(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-cyan"
          />
          <Slider
            label="total"
            value={total}
            min={5000}
            max={50000}
            step={500}
            onChange={(v) => {
              const nt = Math.round(v)
              setTotal(nt)
              if (cursor > nt) setCursor(nt)
              if (warmup > nt) setWarmup(nt)
            }}
            format={(v) => String(Math.round(v))}
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="step" value={String(cursor)} />
            <Readout label="LR" value={curLR.toExponential(2)} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[1fr_220px] gap-4 overflow-hidden">
        <div className="flex flex-col gap-2 min-h-0">
          <div
            ref={boxRef}
            className="relative flex-1 min-h-0"
            onMouseMove={(e) => {
              const r = e.currentTarget.getBoundingClientRect()
              const frac = Math.max(0, Math.min(1, (e.clientX - r.left - 58) / (r.width - 58 - 14)))
              setCursor(Math.round(frac * total))
            }}
          >
            <canvas ref={canvasRef} className="w-full h-full block cursor-crosshair" />
          </div>
        </div>

        <div className="flex flex-col gap-2 min-w-0 overflow-auto">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            schedule
          </div>
          <div className="flex flex-col gap-1">
            {SCHEDULES.map((s) => (
              <button
                key={s.key}
                onClick={() => setKind(s.key)}
                className={cn(
                  'text-left px-2 py-1.5 rounded font-mono text-[11px] transition-all border',
                  kind === s.key
                    ? 'border-current'
                    : 'border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
                style={{ color: kind === s.key ? s.color : undefined }}
              >
                <span className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: s.color }} />
                  {s.name}
                </span>
              </button>
            ))}
          </div>
          <div className="mt-2 font-mono text-[10.5px] leading-relaxed text-dark-text-muted border-t border-dark-border pt-2">
            {sel.desc}
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
