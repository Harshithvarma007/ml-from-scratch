'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Compare FLOPs and memory for pixel-space vs latent-space diffusion.
// Pixel space: H x W x 3, latent: (H/f) x (W/f) x 4 where f is the downscale
// factor (default 8). We approximate per-step cost as:
//   attention:   pixels^2 * d    (self-attention in U-Net)
//   conv:        pixels * k^2 * d^2
//   groupnorm:   pixels * d
// Memory ~ pixels * d * 4 bytes.
// Stacked bars show the breakdown. Sliders for image size and downscale.

function computeCosts(h: number, w: number, channels: number) {
  // Use mid-level U-Net width d (approximate)
  const d = 320
  const pix = h * w
  const conv = pix * 9 * d * d
  const attn = pix * pix * d
  const gn = pix * d
  const total = conv + attn + gn
  const mem = pix * channels * 4
  return { conv, attn, gn, total, mem, pix }
}

const fmtG = (n: number) => {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)} T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} G`
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} M`
  return n.toFixed(0)
}

const fmtMem = (n: number) => {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MB`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`
  return `${n.toFixed(0)} B`
}

export default function PixelVsLatentCost() {
  const [imgSize, setImgSize] = useState(512)
  const [factor, setFactor] = useState(8)

  const pixel = useMemo(() => computeCosts(imgSize, imgSize, 3), [imgSize])
  const latentH = Math.max(1, Math.floor(imgSize / factor))
  const latent = useMemo(
    () => computeCosts(latentH, latentH, 4),
    [latentH],
  )
  const spatialRatio = factor * factor
  const pixelRatio = (imgSize * imgSize) / (latentH * latentH)
  const speedup = pixel.total / Math.max(1, latent.total)

  return (
    <WidgetFrame
      widgetName="PixelVsLatentCost"
      label="diffuse in pixel space vs compressed latent space"
      right={<span className="font-mono">Rombach et al. 2022 — Stable Diffusion's core trick</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="image"
            value={imgSize}
            min={128}
            max={1024}
            step={64}
            onChange={(v) => setImgSize(Math.round(v / 64) * 64)}
            format={(v) => `${Math.round(v)}×${Math.round(v)}`}
            accent="accent-term-purple"
          />
          <Slider
            label="downscale f"
            value={factor}
            min={2}
            max={16}
            step={1}
            onChange={(v) => setFactor(Math.round(v))}
            format={(v) => `${Math.round(v)}×`}
            accent="accent-term-cyan"
          />
          <div className="flex items-center gap-3 ml-auto">
            <Readout
              label="pixels saved"
              value={`${pixelRatio.toFixed(0)}×`}
              accent="text-term-amber"
            />
            <Readout
              label="FLOPs speedup"
              value={`${speedup.toFixed(1)}×`}
              accent="text-term-green"
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-2 gap-4">
        <SidePanel
          title={`pixel space — ${imgSize}×${imgSize}×3`}
          titleColor="text-term-rose"
          cost={pixel}
          max={pixel.total}
        />
        <SidePanel
          title={`latent space — ${latentH}×${latentH}×4`}
          titleColor="text-term-green"
          cost={latent}
          max={pixel.total}
        />
      </div>

      <div className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-dark-bg/90 border border-dark-border rounded px-3 py-2 font-mono text-[10.5px] text-dark-text-muted">
        <div className="text-term-amber text-center mb-1">{spatialRatio}× spatial reduction</div>
        <div className="text-dark-text-secondary text-center">{pixelRatio.toFixed(0)}× fewer pixels · {speedup.toFixed(1)}× fewer FLOPs</div>
      </div>
    </WidgetFrame>
  )
}

function SidePanel({
  title,
  titleColor,
  cost,
  max,
}: {
  title: string
  titleColor: string
  cost: ReturnType<typeof computeCosts>
  max: number
}) {
  const segs = [
    { label: 'attention', val: cost.attn, color: '#a78bfa' },
    { label: 'conv', val: cost.conv, color: '#67e8f9' },
    { label: 'norm', val: cost.gn, color: '#4ade80' },
  ]
  const pct = (v: number) => (v / max) * 100

  return (
    <div className="flex flex-col gap-3 min-w-0 min-h-0">
      <div className={`text-[10px] font-mono uppercase tracking-wider ${titleColor}`}>{title}</div>

      <div className="bg-dark-surface-elevated/40 rounded p-3 font-mono text-[10.5px] leading-relaxed">
        <div className="text-dark-text-disabled uppercase tracking-wider text-[9.5px] mb-1">
          pixel count
        </div>
        <div className="text-dark-text-primary text-[12.5px] tabular-nums mb-2">
          {cost.pix.toLocaleString()}
        </div>
        <div className="text-dark-text-disabled uppercase tracking-wider text-[9.5px] mb-1">
          memory (fp32)
        </div>
        <div className="text-term-amber text-[12px] tabular-nums">
          {fmtMem(cost.mem)}
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <div className="text-[9.5px] font-mono uppercase tracking-wider text-dark-text-disabled">
          FLOPs / step (relative to largest bar)
        </div>
        {segs.map((s) => (
          <div key={s.label} className="flex items-center gap-2">
            <span className="w-16 text-[10px] font-mono text-dark-text-secondary">{s.label}</span>
            <div className="flex-1 h-5 bg-dark-surface-elevated/40 rounded overflow-hidden relative">
              <div className="h-full" style={{ width: `${pct(s.val)}%`, backgroundColor: s.color, opacity: 0.75 }} />
            </div>
            <span className="w-20 text-right font-mono text-[10px] tabular-nums text-dark-text-primary">
              {fmtG(s.val)}
            </span>
          </div>
        ))}
        <div className="flex items-center gap-2 pt-1 border-t border-dark-border">
          <span className="w-16 text-[10px] font-mono text-term-amber">total</span>
          <div className="flex-1 h-6 bg-dark-surface-elevated/40 rounded overflow-hidden relative flex">
            <div style={{ width: `${pct(cost.attn)}%`, backgroundColor: '#a78bfa', opacity: 0.85 }} />
            <div style={{ width: `${pct(cost.conv)}%`, backgroundColor: '#67e8f9', opacity: 0.85 }} />
            <div style={{ width: `${pct(cost.gn)}%`, backgroundColor: '#4ade80', opacity: 0.85 }} />
          </div>
          <span className="w-20 text-right font-mono text-[10px] tabular-nums text-term-amber">
            {fmtG(cost.total)}
          </span>
        </div>
      </div>
    </div>
  )
}
