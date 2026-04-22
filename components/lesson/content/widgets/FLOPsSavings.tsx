'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Compare FLOPs with and without the KV cache to generate N new tokens.
//
// Without cache: each new token re-runs attention over the whole prefix.
// Cost to get the t-th token ≈ c · d · t (the QK and QV dots grow with t).
// Summed over t = 1..N, total ≈ c · d · N·(N+1)/2 → O(N²·d).
//
// With cache: each new token only queries once against the cached keys.
// Cost per token ≈ c · d · S_current, but we reuse keys so per-generation
// it's O(N·d) per token growth, total ≈ c · d · N linearly in N.
//
// We plot both curves and report the speedup at the chosen N, plus a
// wall-clock estimate at 100 TFLOPs.

const D = 4096 // d_model (Llama-7B-ish)
const L = 32 // layers
const FLOPS_PER_ATTN = 4 * D * L // per-token per-step (rough — 4dL accounts for QKV read + output)

function costWithoutCache(N: number): number {
  // quadratic: sum_{t=1..N} FLOPS_PER_ATTN · t = FLOPS_PER_ATTN · N(N+1)/2
  return FLOPS_PER_ATTN * (N * (N + 1)) / 2
}

function costWithCache(N: number): number {
  // linear: sum of O(1) attention work per step, constant in S since K/V are
  // already materialized. We still pay one matmul of length-S per step, but
  // the dominant cost amortizes. Simplify to FLOPS_PER_ATTN · N.
  return FLOPS_PER_ATTN * N
}

function formatFlops(x: number): string {
  if (x >= 1e15) return `${(x / 1e15).toFixed(2)} PFLOP`
  if (x >= 1e12) return `${(x / 1e12).toFixed(2)} TFLOP`
  if (x >= 1e9) return `${(x / 1e9).toFixed(2)} GFLOP`
  if (x >= 1e6) return `${(x / 1e6).toFixed(1)} MFLOP`
  return x.toExponential(1)
}

function formatSeconds(s: number): string {
  if (s >= 60) return `${(s / 60).toFixed(1)} min`
  if (s >= 1) return `${s.toFixed(1)} s`
  if (s >= 1e-3) return `${(s * 1000).toFixed(1)} ms`
  return `${(s * 1e6).toFixed(1)} µs`
}

const HARDWARE_TFLOPS = 100

export default function FLOPsSavings() {
  const [N, setN] = useState(512)

  const wall = useMemo(() => {
    const without = costWithoutCache(N)
    const withc = costWithCache(N)
    return {
      without,
      withc,
      speedup: without / Math.max(withc, 1),
      tWithout: without / (HARDWARE_TFLOPS * 1e12),
      tWith: withc / (HARDWARE_TFLOPS * 1e12),
    }
  }, [N])

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

      const padL = 62
      const padR = 16
      const padT = 14
      const padB = 30
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      const Nmax = 2048
      const yMaxFLOP = costWithoutCache(Nmax)
      const logBounds = (v: number) => Math.max(Math.log10(Math.max(v, 1)), 6)

      const toSx = (n: number) => padL + (n / Nmax) * plotW
      const yMinLog = 6
      const yMaxLog = Math.log10(yMaxFLOP)
      const toSy = (v: number) => padT + plotH - ((logBounds(v) - yMinLog) / (yMaxLog - yMinLog)) * plotH

      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.fillStyle = '#555'
      ctx.textAlign = 'center'
      for (let n = 0; n <= Nmax; n += 512) {
        const sx = toSx(n)
        ctx.beginPath()
        ctx.moveTo(sx, padT)
        ctx.lineTo(sx, padT + plotH)
        ctx.stroke()
        ctx.fillText(String(n), sx, padT + plotH + 14)
      }
      ctx.fillStyle = '#777'
      ctx.fillText('tokens generated (N)', padL + plotW / 2, padT + plotH + 26)
      ctx.textAlign = 'right'
      ctx.fillStyle = '#555'
      for (let li = Math.ceil(yMinLog); li <= yMaxLog; li++) {
        const sy = toSy(Math.pow(10, li))
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(`1e${li}`, padL - 6, sy + 3)
      }

      // Curve: no cache (quadratic)
      ctx.strokeStyle = '#f87171'
      ctx.lineWidth = 2.2
      ctx.beginPath()
      for (let i = 0; i <= 200; i++) {
        const n = (i / 200) * Nmax
        const v = costWithoutCache(Math.max(n, 1))
        const sx = toSx(n)
        const sy = toSy(v)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()

      // Curve: with cache (linear)
      ctx.strokeStyle = '#4ade80'
      ctx.lineWidth = 2.2
      ctx.beginPath()
      for (let i = 0; i <= 200; i++) {
        const n = (i / 200) * Nmax
        const v = costWithCache(Math.max(n, 1))
        const sx = toSx(n)
        const sy = toSy(v)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()

      // Cursor
      const sxN = toSx(N)
      ctx.strokeStyle = 'rgba(255,255,255,0.4)'
      ctx.setLineDash([3, 4])
      ctx.beginPath()
      ctx.moveTo(sxN, padT)
      ctx.lineTo(sxN, padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])

      const syWithout = toSy(costWithoutCache(N))
      const syWith = toSy(costWithCache(N))
      ctx.fillStyle = '#f87171'
      ctx.beginPath()
      ctx.arc(sxN, syWithout, 5, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = '#4ade80'
      ctx.beginPath()
      ctx.arc(sxN, syWith, 5, 0, Math.PI * 2)
      ctx.fill()

      // legend
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.textAlign = 'left'
      ctx.strokeStyle = '#f87171'
      ctx.lineWidth = 2
      ctx.beginPath(); ctx.moveTo(padL + 10, padT + 12); ctx.lineTo(padL + 24, padT + 12); ctx.stroke()
      ctx.fillStyle = '#f87171'
      ctx.fillText('without cache — O(N²·d)', padL + 28, padT + 15)
      ctx.strokeStyle = '#4ade80'
      ctx.beginPath(); ctx.moveTo(padL + 10, padT + 26); ctx.lineTo(padL + 24, padT + 26); ctx.stroke()
      ctx.fillStyle = '#4ade80'
      ctx.fillText('with cache — O(N·d)', padL + 28, padT + 29)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [N])

  return (
    <WidgetFrame
      widgetName="FLOPsSavings"
      label="KV cache speedup — FLOPs to generate N tokens"
      right={<span className="font-mono">d={D} · L={L} · reference: 100 TFLOPs hardware</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="N tokens"
            value={N}
            min={1}
            max={2048}
            step={1}
            onChange={(v) => setN(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="speedup" value={`${wall.speedup.toFixed(1)}×`} accent="text-term-green" />
            <Readout label="no-cache" value={formatFlops(wall.without)} accent="text-term-rose" />
            <Readout label="w/ cache" value={formatFlops(wall.withc)} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[1fr_260px] gap-4 overflow-hidden">
        <div ref={boxRef} className="relative min-h-0">
          <canvas ref={canvasRef} className="w-full h-full block" />
        </div>

        <div className="flex flex-col gap-2 min-w-0 overflow-auto">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            wall clock at {HARDWARE_TFLOPS} TFLOPs
          </div>
          <div className="rounded border border-dark-border bg-dark-surface-elevated/40 p-3">
            <div className="flex items-center justify-between font-mono text-[11px]">
              <span className="text-term-rose">no cache</span>
              <span className="tabular-nums text-dark-text-primary">{formatSeconds(wall.tWithout)}</span>
            </div>
            <div className="flex items-center justify-between font-mono text-[11px] mt-1">
              <span className="text-term-green">with cache</span>
              <span className="tabular-nums text-dark-text-primary">{formatSeconds(wall.tWith)}</span>
            </div>
            <div className="border-t border-dark-border/60 mt-2 pt-2 flex items-center justify-between font-mono text-[11px]">
              <span className="text-dark-text-disabled">you save</span>
              <span className="tabular-nums text-term-amber">{formatSeconds(Math.max(0, wall.tWithout - wall.tWith))}</span>
            </div>
          </div>

          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-1">
            what's actually going on
          </div>
          <div className="font-mono text-[10.5px] leading-relaxed text-dark-text-muted">
            <p>
              without cache, every new token re-projects K and V for every previous token. for token t that's O(t·d) work, and the sum over the full run is the triangular number — hence N².
            </p>
            <p className="mt-1.5">
              with a cache you pay that work exactly once per position, never again. the attention matmul at each step still has length S, but the K,V never get recomputed. total collapses to N per-layer matmuls.
            </p>
            <p className="mt-1.5">
              at N = {N}, no-cache is{' '}
              <span className="text-term-amber tabular-nums">{wall.speedup.toFixed(1)}×</span>{' '}
              slower.
            </p>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
