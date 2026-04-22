'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// KV cache footprint vs sequence length. Formula:
// bytes(seq_len) = 2 · n_layers · n_heads · head_dim · seq_len · bytes_per_fp · batch
// We plot on log-log. Markers at 1k, 4k, 32k, 128k. Slider knobs: model preset,
// precision, batch size.

type Preset = {
  key: string
  name: string
  n_layers: number
  n_heads: number
  head_dim: number
  d_model: number
}

const PRESETS: Preset[] = [
  { key: 'small', name: 'GPT-2 small · 125M', n_layers: 12, n_heads: 12, head_dim: 64, d_model: 768 },
  { key: 'medium', name: 'Llama-7B class', n_layers: 32, n_heads: 32, head_dim: 128, d_model: 4096 },
  { key: 'large', name: 'Llama-70B class', n_layers: 80, n_heads: 64, head_dim: 128, d_model: 8192 },
]

type Precision = 'fp32' | 'fp16' | 'int8'
const BYTES: Record<Precision, number> = { fp32: 4, fp16: 2, int8: 1 }

const MARKER_POINTS = [1024, 4096, 32768, 131072]

function cacheBytes(p: Preset, seq: number, prec: Precision, batch: number): number {
  return 2 * p.n_layers * p.n_heads * p.head_dim * seq * BYTES[prec] * batch
}

function formatBytes(b: number): string {
  if (b >= 1e12) return `${(b / 1e12).toFixed(2)} TB`
  if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`
  if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`
  if (b >= 1e3) return `${(b / 1e3).toFixed(1)} KB`
  return `${b} B`
}

const GPU_80GB = 80 * 1024 ** 3

export default function KVCacheGrowth() {
  const [presetKey, setPresetKey] = useState('medium')
  const [prec, setPrec] = useState<Precision>('fp16')
  const [batch, setBatch] = useState(1)

  const preset = PRESETS.find((p) => p.key === presetKey)!
  const perToken = useMemo(() => cacheBytes(preset, 1, prec, batch), [preset, prec, batch])
  // headroom for weights & activations: assume 40GB free for KV on an 80GB GPU
  const free = GPU_80GB * 0.5
  const maxSeq = Math.floor(free / perToken)

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

      const padL = 70
      const padR = 16
      const padT = 14
      const padB = 32
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const logSeqMin = Math.log10(256)
      const logSeqMax = Math.log10(262144)
      const logBMin = Math.log10(Math.max(cacheBytes(preset, 256, prec, batch), 1))
      const logBMax = Math.log10(Math.max(cacheBytes(preset, 262144, prec, batch), 1))
      const toSx = (ls: number) => padL + ((ls - logSeqMin) / (logSeqMax - logSeqMin)) * plotW
      const toSy = (lb: number) => padT + plotH - ((lb - logBMin) / (logBMax - logBMin)) * plotH

      ctx.font = '10px "JetBrains Mono", monospace'
      // grid
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.fillStyle = '#555'
      ctx.textAlign = 'center'
      for (let li = Math.ceil(logSeqMin); li <= logSeqMax; li += 1) {
        const s = Math.pow(10, li)
        const sx = toSx(li)
        ctx.beginPath()
        ctx.moveTo(sx, padT)
        ctx.lineTo(sx, padT + plotH)
        ctx.stroke()
        const lbl = s >= 1e6 ? `${s / 1e6}M` : s >= 1e3 ? `${s / 1e3}k` : String(s)
        ctx.fillText(lbl, sx, padT + plotH + 14)
      }
      ctx.fillStyle = '#777'
      ctx.fillText('seq length (log)', padL + plotW / 2, padT + plotH + 26)

      ctx.textAlign = 'right'
      ctx.fillStyle = '#555'
      for (let li = Math.ceil(logBMin); li <= logBMax; li += 1) {
        const b = Math.pow(10, li)
        const sy = toSy(li)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(formatBytes(b), padL - 6, sy + 3)
      }

      // 80 GB line
      if (Math.log10(GPU_80GB) >= logBMin && Math.log10(GPU_80GB) <= logBMax) {
        const sy80 = toSy(Math.log10(GPU_80GB))
        ctx.strokeStyle = '#f87171'
        ctx.setLineDash([4, 4])
        ctx.beginPath()
        ctx.moveTo(padL, sy80)
        ctx.lineTo(padL + plotW, sy80)
        ctx.stroke()
        ctx.setLineDash([])
        ctx.fillStyle = '#f87171'
        ctx.textAlign = 'left'
        ctx.fillText('80 GB', padL + plotW - 40, sy80 - 4)
      }

      // Line
      ctx.strokeStyle = '#67e8f9'
      ctx.lineWidth = 2.2
      ctx.beginPath()
      for (let i = 0; i <= 200; i++) {
        const ls = logSeqMin + (i / 200) * (logSeqMax - logSeqMin)
        const s = Math.pow(10, ls)
        const b = cacheBytes(preset, s, prec, batch)
        const lb = Math.log10(Math.max(b, 1))
        const sx = toSx(ls)
        const sy = toSy(lb)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()

      // Markers
      MARKER_POINTS.forEach((s) => {
        const b = cacheBytes(preset, s, prec, batch)
        const sx = toSx(Math.log10(s))
        const sy = toSy(Math.log10(Math.max(b, 1)))
        ctx.fillStyle = '#fbbf24'
        ctx.beginPath()
        ctx.arc(sx, sy, 4.5, 0, Math.PI * 2)
        ctx.fill()
        ctx.fillStyle = '#fbbf24'
        ctx.textAlign = 'left'
        ctx.fillText(`${s >= 1024 ? `${s / 1024}k` : s} → ${formatBytes(b)}`, sx + 8, sy + 3)
      })
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [preset, prec, batch])

  return (
    <WidgetFrame
      widgetName="KVCacheGrowth"
      label="KV cache growth — bytes vs sequence length"
      right={<span className="font-mono">{preset.name} · L={preset.n_layers} · H={preset.n_heads} · d_h={preset.head_dim}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            {PRESETS.map((p) => (
              <button
                key={p.key}
                onClick={() => setPresetKey(p.key)}
                className={cn(
                  'px-2.5 py-1 rounded text-[10.5px] font-mono uppercase transition-all',
                  presetKey === p.key
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {p.key}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1.5">
            {(['fp32', 'fp16', 'int8'] as Precision[]).map((p) => (
              <button
                key={p}
                onClick={() => setPrec(p)}
                className={cn(
                  'px-2 py-1 rounded text-[10.5px] font-mono transition-all',
                  prec === p
                    ? 'bg-term-amber/20 text-term-amber border border-term-amber/40'
                    : 'border border-dark-border text-dark-text-secondary',
                )}
              >
                {p}
              </button>
            ))}
          </div>
          <Slider label="batch" value={batch} min={1} max={32} step={1} onChange={(v) => setBatch(Math.round(v))} format={(v) => String(Math.round(v))} accent="accent-term-cyan" />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="per token" value={formatBytes(perToken)} accent="text-term-amber" />
            <Readout label="max seq @ 80GB/2" value={maxSeq.toLocaleString('en-US')} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[1fr_240px] gap-4 overflow-hidden">
        <div ref={boxRef} className="relative min-h-0">
          <canvas ref={canvasRef} className="w-full h-full block" />
        </div>

        <div className="flex flex-col gap-2 min-w-0 overflow-auto">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            formula
          </div>
          <div className="rounded border border-dark-border bg-dark-surface-elevated/40 p-2.5 font-mono text-[10px] leading-relaxed text-dark-text-muted">
            <div className="text-term-cyan break-words">
              bytes = 2 · L · H · d_h · S · fp · B
            </div>
            <div className="mt-1 text-dark-text-disabled">
              factor 2 = K and V, both stored.
            </div>
          </div>

          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-1">
            milestones at current config
          </div>
          <div className="flex flex-col gap-1 font-mono text-[10px]">
            {MARKER_POINTS.map((s) => {
              const b = cacheBytes(preset, s, prec, batch)
              const fits = b < GPU_80GB
              return (
                <div key={s} className="flex items-center justify-between px-2 py-1 rounded bg-dark-surface-elevated/30">
                  <span className="text-dark-text-secondary">
                    {s >= 1024 ? `${s / 1024}k` : s} tokens
                  </span>
                  <span className={cn('tabular-nums', fits ? 'text-term-green' : 'text-term-rose')}>
                    {formatBytes(b)}
                  </span>
                </div>
              )
            })}
          </div>

          <div className="mt-1 font-mono text-[10.5px] leading-relaxed text-dark-text-muted border-t border-dark-border pt-2">
            the per-token cost scales linearly; halving precision halves the bill. batch amplifies the total — 32 parallel requests at 4k tokens can tip a 7B model off an 80GB card.
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
