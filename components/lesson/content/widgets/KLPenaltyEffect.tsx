'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Two canvas plots stacked:
//  top: the reference policy distribution (fixed narrow Gaussian) vs the
//       "optimized" policy distribution. As β → 0 the policy drifts toward
//       a spikier, higher-reward mode (simulated with a target). As β grows
//       the policy is pulled back toward the reference.
//  bottom: the plot of expected reward as a function of β, showing the
//       classic inverted-U: too little KL and the policy explodes into reward
//       hacking (low proxy reward); too much and it can't improve at all.

function dnorm(x: number, mu: number, sigma: number): number {
  return Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI))
}

// Our "reference" π_ref: N(0, 1).
// The optimizer's target (peaked at reward mode): N(1.5, 0.7).
// As β grows, policy σ widens and μ shifts back to 0.
function policyParams(beta: number): { mu: number; sigma: number } {
  // Simple closed-ish interpolation. At β=0 the policy is peaked at reward target (μ=1.5).
  // At β=0.5 it's close to reference (μ→0).
  const t = Math.min(1, beta / 0.4)
  const mu = 1.5 * (1 - t) * 1.1 // slight overshoot at very low β
  const sigma = 0.7 + (1 - 1 / (1 + 2 * beta)) * 0.3
  return { mu, sigma }
}

// Expected reward shape: rises fast as β→0 because policy peaks near
// reward target, but at very low β reward hacking causes a crash. Peak
// around β ≈ 0.05.
function rewardCurve(beta: number): number {
  // Simulated: -exp((β-0.05)²/2σ²) style isn't ideal — use a hand shape.
  if (beta <= 0) return 0.1
  const peak = 0.05
  // Below peak: reward collapses due to reward hacking.
  if (beta < peak) {
    const t = beta / peak
    return 0.4 + t * 0.55 // climbs from 0.4 to 0.95 at the peak
  }
  // Above peak: slow decay as policy is pulled back toward SFT.
  const over = beta - peak
  return 0.95 * Math.exp(-over * 5) * (1 + Math.sin(over * 8) * 0.02)
}

// KL divergence for Gaussians: KL(p‖q) = log(σ_q/σ_p) + (σ_p² + (μ_p-μ_q)²)/(2σ_q²) - 1/2
function klGaussian(mu1: number, sig1: number, mu2: number, sig2: number): number {
  return (
    Math.log(sig2 / sig1) +
    (sig1 * sig1 + (mu1 - mu2) ** 2) / (2 * sig2 * sig2) -
    0.5
  )
}

export default function KLPenaltyEffect() {
  const [beta, setBeta] = useState(0.05)
  const topRef = useRef<HTMLCanvasElement | null>(null)
  const botRef = useRef<HTMLCanvasElement | null>(null)
  const topBox = useRef<HTMLDivElement | null>(null)
  const botBox = useRef<HTMLDivElement | null>(null)

  const { mu, sigma } = useMemo(() => policyParams(beta), [beta])
  const kl = useMemo(() => klGaussian(mu, sigma, 0, 1), [mu, sigma])
  const reward = useMemo(() => rewardCurve(beta), [beta])
  const effObjective = reward - beta * kl

  useEffect(() => {
    const canvas = topRef.current
    const box = topBox.current
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

      const padL = 32
      const padR = 14
      const padT = 14
      const padB = 20
      const plotW = w - padL - padR
      const plotH = h - padT - padB
      const xMin = -3
      const xMax = 4
      const toSx = (x: number) => padL + ((x - xMin) / (xMax - xMin)) * plotW
      const yMax = 0.6
      const toSy = (y: number) => padT + plotH - (y / yMax) * plotH

      // Grid
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.fillStyle = '#555'
      ctx.font = '9px "JetBrains Mono", monospace'
      ctx.textAlign = 'right'
      ;[0, 0.2, 0.4, 0.6].forEach((y) => {
        ctx.beginPath()
        ctx.moveTo(padL, toSy(y))
        ctx.lineTo(padL + plotW, toSy(y))
        ctx.stroke()
        ctx.fillText(y.toFixed(1), padL - 4, toSy(y) + 3)
      })
      ctx.textAlign = 'center'
      ;[-3, -1, 0, 1, 2, 3, 4].forEach((x) =>
        ctx.fillText(String(x), toSx(x), padT + plotH + 13),
      )

      // π_ref: N(0, 1) (amber, filled)
      ctx.fillStyle = 'rgba(251, 191, 36, 0.25)'
      ctx.strokeStyle = '#fbbf24'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      for (let i = 0; i <= 200; i++) {
        const x = xMin + (i / 200) * (xMax - xMin)
        const y = dnorm(x, 0, 1)
        const sx = toSx(x)
        const sy = toSy(y)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()
      ctx.lineTo(toSx(xMax), toSy(0))
      ctx.lineTo(toSx(xMin), toSy(0))
      ctx.closePath()
      ctx.fill()

      // π_θ: policy (cyan)
      ctx.fillStyle = 'rgba(103, 232, 249, 0.2)'
      ctx.strokeStyle = '#67e8f9'
      ctx.lineWidth = 1.8
      ctx.beginPath()
      for (let i = 0; i <= 200; i++) {
        const x = xMin + (i / 200) * (xMax - xMin)
        const y = dnorm(x, mu, sigma)
        const sx = toSx(x)
        const sy = toSy(y)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()
      ctx.lineTo(toSx(xMax), toSy(0))
      ctx.lineTo(toSx(xMin), toSy(0))
      ctx.closePath()
      ctx.fill()

      // Reward target region shading
      ctx.fillStyle = 'rgba(74, 222, 128, 0.1)'
      ctx.fillRect(toSx(1), padT, toSx(2) - toSx(1), plotH)
      ctx.fillStyle = '#4ade80'
      ctx.font = '9.5px "JetBrains Mono", monospace'
      ctx.textAlign = 'left'
      ctx.fillText('reward peak', toSx(1) + 4, padT + 12)

      // Legend
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#fbbf24'
      ctx.textAlign = 'left'
      ctx.fillText('π_ref (frozen SFT)', padL + 4, padT + 12)
      ctx.fillStyle = '#67e8f9'
      ctx.fillText('π_θ (current policy)', padL + 4, padT + 26)

      // KL readout on plot
      ctx.fillStyle = '#f472b6'
      ctx.textAlign = 'right'
      ctx.fillText(`KL = ${kl.toFixed(3)}`, padL + plotW - 4, padT + 12)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [mu, sigma, kl])

  useEffect(() => {
    const canvas = botRef.current
    const box = botBox.current
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

      const padL = 32
      const padR = 14
      const padT = 14
      const padB = 22
      const plotW = w - padL - padR
      const plotH = h - padT - padB
      const xMin = 0
      const xMax = 0.5
      const toSx = (x: number) => padL + ((x - xMin) / (xMax - xMin)) * plotW
      const toSy = (y: number) => padT + plotH - (y / 1.0) * plotH

      // Grid
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.fillStyle = '#555'
      ctx.font = '9px "JetBrains Mono", monospace'
      ctx.textAlign = 'right'
      ;[0, 0.25, 0.5, 0.75, 1].forEach((y) => {
        ctx.beginPath()
        ctx.moveTo(padL, toSy(y))
        ctx.lineTo(padL + plotW, toSy(y))
        ctx.stroke()
        ctx.fillText(y.toFixed(2), padL - 4, toSy(y) + 3)
      })
      ctx.textAlign = 'center'
      ;[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5].forEach((x) =>
        ctx.fillText(x.toFixed(2), toSx(x), padT + plotH + 13),
      )

      // Reward curve (green)
      ctx.strokeStyle = '#4ade80'
      ctx.lineWidth = 2
      ctx.beginPath()
      for (let i = 0; i <= 200; i++) {
        const b = xMin + (i / 200) * (xMax - xMin)
        const r = rewardCurve(b)
        const sx = toSx(b)
        const sy = toSy(r)
        if (i === 0) ctx.moveTo(sx, sy)
        else ctx.lineTo(sx, sy)
      }
      ctx.stroke()

      // Sweet spot marker
      const peak = 0.05
      const sxP = toSx(peak)
      ctx.strokeStyle = 'rgba(251, 191, 36, 0.75)'
      ctx.setLineDash([3, 4])
      ctx.beginPath()
      ctx.moveTo(sxP, padT)
      ctx.lineTo(sxP, padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = '#fbbf24'
      ctx.font = '9.5px "JetBrains Mono", monospace'
      ctx.textAlign = 'left'
      ctx.fillText('sweet spot β≈0.05', sxP + 4, padT + 12)

      // Current β cursor
      const sxB = toSx(beta)
      ctx.strokeStyle = 'rgba(103, 232, 249, 0.9)'
      ctx.setLineDash([2, 3])
      ctx.beginPath()
      ctx.moveTo(sxB, padT)
      ctx.lineTo(sxB, padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])
      const r = rewardCurve(beta)
      ctx.fillStyle = '#67e8f9'
      ctx.beginPath()
      ctx.arc(sxB, toSy(r), 4, 0, Math.PI * 2)
      ctx.fill()

      // Region labels
      ctx.fillStyle = 'rgba(248, 113, 113, 0.6)'
      ctx.font = '9.5px "JetBrains Mono", monospace'
      ctx.textAlign = 'center'
      ctx.fillText('reward hacking', toSx(0.015), padT + plotH - 6)
      ctx.fillStyle = 'rgba(167, 139, 250, 0.6)'
      ctx.fillText('over-regularized', toSx(0.35), padT + plotH - 6)

      // Title
      ctx.fillStyle = '#888'
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.textAlign = 'left'
      ctx.fillText('E[reward] vs β', padL + 4, padT + 12)
      ctx.textAlign = 'center'
      ctx.fillText('β (KL coefficient)', padL + plotW / 2, padT + plotH + 20)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [beta])

  return (
    <WidgetFrame
      widgetName="KLPenaltyEffect"
      label="KL penalty — the leash between policy and reference"
      right={<span className="font-mono">loss = E[r] − β · KL(π_θ ‖ π_ref)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="β"
            value={beta}
            min={0}
            max={0.5}
            step={0.005}
            onChange={setBeta}
            format={(v) => v.toFixed(3)}
            accent="accent-term-purple"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="KL" value={kl.toFixed(3)} accent="text-term-pink" />
            <Readout label="E[r]" value={reward.toFixed(3)} accent="text-term-green" />
            <Readout label="obj − β·KL" value={effObjective.toFixed(3)} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-2 grid grid-rows-[1fr_1fr] gap-1 overflow-hidden">
        <div ref={topBox} className="relative min-h-0">
          <canvas ref={topRef} className="w-full h-full block" />
        </div>
        <div ref={botBox} className="relative min-h-0">
          <canvas ref={botRef} className="w-full h-full block" />
        </div>
      </div>
    </WidgetFrame>
  )
}
