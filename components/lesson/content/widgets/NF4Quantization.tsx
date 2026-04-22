'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// NF4 quantization: 16 bucket boundaries placed at quantiles of the normal
// distribution, so each bucket contains ~6.25% of weights. Show the weight
// histogram, overlay bucket edges, and draw the quantization error as a
// secondary histogram. Blocksize controls how many weights share a scale —
// smaller blocks mean less error but more metadata bytes per weight.

function erf(x: number): number {
  const sign = Math.sign(x)
  const a = Math.abs(x)
  const t = 1 / (1 + 0.3275911 * a)
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-a * a)
  return sign * y
}

function phi(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2))
}

// Inverse CDF of standard normal (Beasley-Springer-Moro or rational approx).
function invPhi(p: number): number {
  if (p <= 0) return -6
  if (p >= 1) return 6
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239]
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1]
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416]
  const pLow = 0.02425
  const pHigh = 1 - pLow
  let q: number, r: number
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p))
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  }
  if (p <= pHigh) {
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  }
  q = Math.sqrt(-2 * Math.log(1 - p))
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
}

// NF4 quantile midpoints — 16 levels placed so each step covers 1/16 of mass.
const NF4_EDGES: number[] = Array.from({ length: 17 }, (_, i) => invPhi(i / 16))
const NF4_CENTERS: number[] = Array.from({ length: 16 }, (_, i) => {
  const lo = NF4_EDGES[i]
  const hi = NF4_EDGES[i + 1]
  const safeLo = isFinite(lo) ? lo : -3
  const safeHi = isFinite(hi) ? hi : 3
  return (safeLo + safeHi) / 2
})

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gauss(rng: () => number): number {
  const u = Math.max(rng(), 1e-9)
  const v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function quantizeNF4(v: number): number {
  // binary search into NF4_CENTERS; return nearest center
  let best = 0
  let bestDist = Infinity
  for (let i = 0; i < NF4_CENTERS.length; i++) {
    const d = Math.abs(NF4_CENTERS[i] - v)
    if (d < bestDist) {
      bestDist = d
      best = i
    }
  }
  return NF4_CENTERS[best]
}

export default function NF4Quantization() {
  const [blocksize, setBlocksize] = useState(64)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const sample = useMemo(() => {
    const rng = mulberry32(7)
    const N = 8000
    const out: number[] = []
    for (let i = 0; i < N; i++) out.push(gauss(rng))
    return out
  }, [])

  const { mse, bitsPerWeight, compression } = useMemo(() => {
    // Simulate per-block scale: normalize by absmax within block.
    let sse = 0
    for (let b = 0; b < sample.length; b += blocksize) {
      const blk = sample.slice(b, b + blocksize)
      const scale = Math.max(...blk.map((v) => Math.abs(v)), 1e-9)
      for (const v of blk) {
        const norm = v / scale
        const q = quantizeNF4(norm) * scale
        sse += (q - v) ** 2
      }
    }
    const mseVal = sse / sample.length
    // Bits per weight: 4 bits + (metadata). Absmax scale (fp16) = 16 bits per block.
    const bits = 4 + 16 / blocksize
    const comp = 16 / bits
    return { mse: mseVal, bitsPerWeight: bits, compression: comp }
  }, [sample, blocksize])

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

      const padL = 36
      const padR = 16
      const padT = 14
      const padB = 24
      const plotW = w - padL - padR
      // Top: histogram of weights. Bottom: quantization error.
      const splitY = padT + (h - padT - padB) * 0.62
      const topH = splitY - padT
      const bottomH = h - padB - splitY - 6

      // Weight histogram (rebuild with scale normalization)
      const bins = 60
      const xMin = -3.2
      const xMax = 3.2
      const toSx = (x: number) => padL + ((x - xMin) / (xMax - xMin)) * plotW
      const hist = new Array(bins).fill(0)
      const errHist = new Array(bins).fill(0)
      const errCount = new Array(bins).fill(0)
      for (let b = 0; b < sample.length; b += blocksize) {
        const blk = sample.slice(b, b + blocksize)
        const scale = Math.max(...blk.map((v) => Math.abs(v)), 1e-9)
        for (const v of blk) {
          const binI = Math.min(bins - 1, Math.max(0, Math.floor(((v - xMin) / (xMax - xMin)) * bins)))
          hist[binI]++
          const norm = v / scale
          const q = quantizeNF4(norm) * scale
          errHist[binI] += (q - v) ** 2
          errCount[binI]++
        }
      }

      // Normalize
      const histMax = Math.max(...hist)
      ctx.fillStyle = 'rgba(103, 232, 249, 0.35)'
      for (let i = 0; i < bins; i++) {
        const x0 = toSx(xMin + (i / bins) * (xMax - xMin))
        const x1 = toSx(xMin + ((i + 1) / bins) * (xMax - xMin))
        const barH = (hist[i] / histMax) * (topH - 8)
        ctx.fillRect(x0, splitY - barH, x1 - x0 - 1, barH)
      }

      // Theoretical PDF overlay (scaled)
      ctx.strokeStyle = '#67e8f9'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      const pdfScale = (histMax / (sample.length / bins)) * (sample.length * (xMax - xMin)) / bins
      for (let i = 0; i <= 200; i++) {
        const x = xMin + (i / 200) * (xMax - xMin)
        const y = Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)
        const sx = toSx(x)
        const sy = splitY - (y * pdfScale / histMax) * (topH - 8)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()

      // NF4 edges (vertical dashed)
      ctx.strokeStyle = 'rgba(251, 191, 36, 0.7)'
      ctx.setLineDash([2, 3])
      ctx.lineWidth = 1
      NF4_EDGES.forEach((e) => {
        if (!isFinite(e) || e < xMin || e > xMax) return
        const sx = toSx(e)
        ctx.beginPath()
        ctx.moveTo(sx, padT)
        ctx.lineTo(sx, splitY)
        ctx.stroke()
      })
      // NF4 centers
      ctx.setLineDash([])
      ctx.fillStyle = '#fbbf24'
      NF4_CENTERS.forEach((c) => {
        if (c < xMin || c > xMax) return
        const sx = toSx(c)
        ctx.beginPath()
        ctx.arc(sx, splitY - 2, 2.5, 0, Math.PI * 2)
        ctx.fill()
      })

      // x axis label
      ctx.fillStyle = '#777'
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.textAlign = 'center'
      ctx.fillText('weight value (std-normalized)', padL + plotW / 2, splitY + 12)

      // Bottom: squared error histogram
      const errPerBin = errHist.map((s, i) => (errCount[i] > 0 ? s / errCount[i] : 0))
      const errMax = Math.max(...errPerBin, 1e-6)
      ctx.fillStyle = 'rgba(244, 114, 182, 0.5)'
      for (let i = 0; i < bins; i++) {
        const x0 = toSx(xMin + (i / bins) * (xMax - xMin))
        const x1 = toSx(xMin + ((i + 1) / bins) * (xMax - xMin))
        const barH = (errPerBin[i] / errMax) * bottomH
        ctx.fillRect(x0, splitY + 6 + (bottomH - barH), x1 - x0 - 1, barH)
      }

      // Split line
      ctx.strokeStyle = '#2a2a32'
      ctx.beginPath()
      ctx.moveTo(padL, splitY + 3)
      ctx.lineTo(padL + plotW, splitY + 3)
      ctx.stroke()

      // Labels
      ctx.fillStyle = '#67e8f9'
      ctx.textAlign = 'left'
      ctx.fillText('weights + NF4 grid', padL + 4, padT + 10)
      ctx.fillStyle = '#f472b6'
      ctx.fillText('squared error', padL + 4, splitY + 18)

      // y axis labels
      ctx.fillStyle = '#555'
      ctx.textAlign = 'right'
      ctx.fillText('0', padL - 6, splitY - 2)
      ctx.fillText('0', padL - 6, splitY + 6 + bottomH - 2)
      ctx.fillText(errMax.toExponential(1), padL - 6, splitY + 16)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [sample, blocksize])

  return (
    <WidgetFrame
      widgetName="NF4Quantization"
      label="NF4 quantization — quantile buckets over a normal"
      right={<span className="font-mono">16 levels = 4 bits · edges at Φ⁻¹(k/16)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="block size"
            value={blocksize}
            min={32}
            max={128}
            step={32}
            onChange={(v) => setBlocksize(v as 32 | 64 | 128)}
            format={(v) => String(v)}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="MSE" value={mse.toExponential(2)} accent={mse < 0.01 ? 'text-term-green' : 'text-term-rose'} />
            <Readout label="bits/weight" value={bitsPerWeight.toFixed(2)} accent="text-term-cyan" />
            <Readout label="compression" value={`${compression.toFixed(2)}×`} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div ref={boxRef} className="absolute inset-0 p-2">
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>
    </WidgetFrame>
  )
}
