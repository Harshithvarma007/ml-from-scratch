'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Six training-curve presets. Clicking a card loads that curve into the big
// plot with a diagnosis and a prescribed fix. The curves are generated
// procedurally from believable models of each failure — no recorded data,
// just closed-form decay / oscillation / divergence that match what you'd
// actually see in a TensorBoard log.

type Preset = {
  id: string
  name: string
  symptom: string
  diagnosis: string
  prescription: string
  accent: string
  generator: (step: number, rng: () => number) => { train: number; val: number }
}

const STEPS = 600

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

const PRESETS: Preset[] = [
  {
    id: 'healthy',
    name: 'healthy convergence',
    symptom: 'smooth log-linear descent, val tracks train',
    diagnosis: 'Textbook. Loss decays roughly as 1/t, noise is small relative to signal, val stays just above train.',
    prescription: 'Ship it. Log anyway in case something regresses next epoch.',
    accent: '#4ade80',
    generator: (step, rng) => {
      const t = step / STEPS
      const base = 2.3 * Math.exp(-3.2 * t) + 0.18
      const noise = 0.02 * gauss(rng)
      return {
        train: Math.max(0.01, base + noise),
        val: Math.max(0.01, base + 0.04 + 0.025 * gauss(rng)),
      }
    },
  },
  {
    id: 'divergence',
    name: 'divergence (NaN)',
    symptom: 'loss climbs, then explodes into infinity',
    diagnosis: 'Learning rate above the stability threshold, or a gradient explosion no clipping caught. One batch flipped a weight by ten orders of magnitude and the model never recovered.',
    prescription: 'Drop LR by 10x. Add grad-norm clipping (typical: 1.0). Check for fp16 overflow on large activations.',
    accent: '#f87171',
    generator: (step, rng) => {
      const t = step / STEPS
      if (t < 0.35) {
        const base = 2.3 * Math.exp(-2.0 * t) + 0.4
        return { train: base + 0.02 * gauss(rng), val: base + 0.06 + 0.03 * gauss(rng) }
      }
      const blow = Math.pow(1.06, (t - 0.35) * STEPS)
      const train = Math.min(50, 1.2 + blow * 0.015 + 0.05 * gauss(rng))
      const val = Math.min(50, 1.3 + blow * 0.018 + 0.05 * gauss(rng))
      return { train, val }
    },
  },
  {
    id: 'plateau',
    name: 'plateau (too-low LR)',
    symptom: 'loss drops a hair, then goes flat forever',
    diagnosis: 'LR is so small each step barely moves the weights. Or the model is sitting on a saddle / wide plateau and there is no curvature to push it off. Also possible: dead ReLUs or a bug zeroing gradients.',
    prescription: 'Increase LR by 3-10x. Try a warmup + cosine schedule. Audit for dead neurons and zero-gradient paths.',
    accent: '#67e8f9',
    generator: (step, rng) => {
      const t = step / STEPS
      const base = 2.3 - 0.5 * (1 - Math.exp(-6 * t)) + 0.01 * gauss(rng)
      return {
        train: Math.max(1.7, base),
        val: Math.max(1.72, base + 0.03 + 0.012 * gauss(rng)),
      }
    },
  },
  {
    id: 'spiky',
    name: 'spiky (too-high LR)',
    symptom: 'downward trend but wildly jagged',
    diagnosis: 'LR is above the noise-tolerance threshold. Each step is a gamble; bad batches knock the weights around enough to un-do progress. Rough convergence is still happening but training is unstable.',
    prescription: 'Halve the LR. Use EMA / Lookahead to smooth. Larger batch size lowers gradient variance proportionally.',
    accent: '#fbbf24',
    generator: (step, rng) => {
      const t = step / STEPS
      const base = 2.3 * Math.exp(-2.6 * t) + 0.35
      const spike = 0.35 * gauss(rng) * (0.6 + 0.4 * Math.sin(step * 0.3))
      return {
        train: Math.max(0.1, base + spike),
        val: Math.max(0.1, base + 0.08 + 0.3 * gauss(rng)),
      }
    },
  },
  {
    id: 'overfit',
    name: 'overfitting',
    symptom: 'train drops, val turns and climbs',
    diagnosis: 'The model has enough capacity to memorize train-set quirks, and the dataset is too small to punish it. The gap between train and val is growing — the signature of memorization.',
    prescription: 'Add dropout / weight decay / augmentation. Early-stop at val minimum. Expand the dataset. Shrink the model.',
    accent: '#f472b6',
    generator: (step, rng) => {
      const t = step / STEPS
      const train = Math.max(0.02, 2.3 * Math.exp(-4.5 * t) + 0.05 + 0.015 * gauss(rng))
      let val
      if (t < 0.35) {
        val = 2.3 * Math.exp(-3.0 * t) + 0.25 + 0.02 * gauss(rng)
      } else {
        const rising = 0.6 + 0.7 * (t - 0.35)
        val = rising + 0.025 * gauss(rng)
      }
      return { train, val: Math.max(0.05, val) }
    },
  },
  {
    id: 'double-descent',
    name: 'double descent',
    symptom: 'val loss dips, climbs, then drops again',
    diagnosis: 'Classic interpolation-threshold behavior. Around the point where the model crosses from under- to over-parameterized, val loss peaks. Past that, extra capacity actually helps generalization.',
    prescription: 'Train longer / bigger. Counter-intuitively: the fix for an overparameterized model is more overparameterization.',
    accent: '#a78bfa',
    generator: (step, rng) => {
      const t = step / STEPS
      const train = Math.max(0.02, 2.3 * Math.exp(-5.5 * t) + 0.05 + 0.01 * gauss(rng))
      let val
      if (t < 0.22) {
        val = 2.3 * Math.exp(-3.2 * t) + 0.25
      } else if (t < 0.5) {
        val = 0.6 + 0.85 * Math.sin(((t - 0.22) / 0.28) * Math.PI * 0.5)
      } else {
        val = 1.45 * Math.exp(-2.2 * (t - 0.5)) * 0.5 + 0.25
      }
      return { train, val: Math.max(0.1, val + 0.02 * gauss(rng)) }
    },
  },
]

function generate(p: Preset): { train: number[]; val: number[] } {
  const rng = mulberry32(7)
  const train: number[] = []
  const val: number[] = []
  for (let s = 0; s < STEPS; s++) {
    const out = p.generator(s, rng)
    train.push(out.train)
    val.push(out.val)
  }
  return { train, val }
}

export default function LossCurveGallery() {
  const [active, setActive] = useState<string>('healthy')
  const preset = PRESETS.find((p) => p.id === active)!
  const curves = useMemo(() => generate(preset), [preset])

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

      const padL = 52
      const padR = 14
      const padT = 22
      const padB = 32
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      // Log-y between 0.01 and 50.
      const yMinLog = Math.log10(0.01)
      const yMaxLog = Math.log10(50)

      const toSx = (s: number) => padL + (s / (STEPS - 1)) * plotW
      const toSy = (v: number) => {
        const clipped = Math.max(0.01, Math.min(50, v))
        const lv = Math.log10(clipped)
        return padT + plotH - ((lv - yMinLog) / (yMaxLog - yMinLog)) * plotH
      }

      // Grid + labels
      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.fillStyle = '#555'
      ctx.strokeStyle = '#181818'
      ctx.lineWidth = 1
      ctx.textAlign = 'right'
      ;[0.01, 0.1, 1, 10].forEach((v) => {
        const sy = toSy(v)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(String(v), padL - 6, sy + 3)
      })
      ctx.textAlign = 'center'
      ;[0, 150, 300, 450, 600].forEach((s) => ctx.fillText(String(s), toSx(Math.min(s, STEPS - 1)), padT + plotH + 14))
      ctx.fillStyle = '#777'
      ctx.fillText('step', padL + plotW / 2, padT + plotH + 26)
      ctx.save()
      ctx.translate(padL - 38, padT + plotH / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.textAlign = 'center'
      ctx.fillText('loss (log)', 0, 0)
      ctx.restore()

      // Draw val first (behind), then train.
      const drawSeries = (data: number[], color: string, alpha: number, width: number) => {
        ctx.strokeStyle = color
        ctx.globalAlpha = alpha
        ctx.lineWidth = width
        ctx.beginPath()
        data.forEach((v, s) => {
          const sx = toSx(s)
          const sy = toSy(v)
          if (s === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        ctx.stroke()
        ctx.globalAlpha = 1
      }
      drawSeries(curves.val, '#f472b6', 0.9, 1.6)
      drawSeries(curves.train, preset.accent, 1, 2)

      // Legend
      ctx.textAlign = 'left'
      ctx.font = '11px "JetBrains Mono", monospace'
      ctx.fillStyle = preset.accent
      ctx.fillRect(padL + 8, padT + 6, 12, 2)
      ctx.fillStyle = '#ccc'
      ctx.fillText('train', padL + 24, padT + 10)
      ctx.fillStyle = '#f472b6'
      ctx.fillRect(padL + 70, padT + 6, 12, 2)
      ctx.fillStyle = '#ccc'
      ctx.fillText('val', padL + 86, padT + 10)

      // Title watermark (top right) — preset name
      ctx.fillStyle = '#333'
      ctx.textAlign = 'right'
      ctx.font = 'bold 14px "JetBrains Mono", monospace'
      ctx.fillText(preset.name, padL + plotW - 6, padT + 14)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [curves, preset])

  const finalTrain = curves.train[curves.train.length - 1]
  const finalVal = curves.val[curves.val.length - 1]
  const gap = finalVal - finalTrain

  return (
    <WidgetFrame
      widgetName="LossCurveGallery"
      label="loss curve gallery — six failure modes, one plot"
      right={<span>click a card to load · log-y</span>}
      aspect="tall"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-3 ml-auto">
            <Readout
              label="train"
              value={isFinite(finalTrain) ? finalTrain.toFixed(3) : '∞'}
              accent="text-term-amber"
            />
            <Readout
              label="val"
              value={isFinite(finalVal) ? finalVal.toFixed(3) : '∞'}
              accent="text-term-pink"
            />
            <Readout
              label="gap"
              value={isFinite(gap) ? gap.toFixed(3) : '∞'}
              accent={gap > 0.3 ? 'text-term-rose' : 'text-term-green'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 flex flex-col">
        <div className="flex-1 min-h-0 relative" ref={boxRef}>
          <canvas ref={canvasRef} className="w-full h-full block" />
        </div>
        <div className="border-t border-dark-border bg-dark-surface-elevated/30 px-3 py-2.5">
          <div className="grid grid-cols-3 gap-2 mb-2.5">
            {PRESETS.map((p) => {
              const isActive = p.id === active
              return (
                <button
                  key={p.id}
                  onClick={() => setActive(p.id)}
                  className={cn(
                    'text-left border rounded-md px-2.5 py-1.5 transition-all',
                    isActive
                      ? 'border-dark-border-hover bg-dark-surface-elevated'
                      : 'border-dark-border bg-dark-bg hover:border-dark-border-hover hover:bg-dark-surface-elevated/60'
                  )}
                  style={isActive ? { borderColor: p.accent } : undefined}
                >
                  <div
                    className="text-[11px] font-mono uppercase tracking-wider mb-0.5"
                    style={{ color: p.accent }}
                  >
                    {p.name}
                  </div>
                  <div className="text-[10.5px] text-dark-text-muted font-sans leading-snug">
                    {p.symptom}
                  </div>
                </button>
              )
            })}
          </div>
          <div className="border-t border-dark-border/50 pt-2">
            <div className="flex items-baseline gap-2 mb-1">
              <span
                className="text-[10px] font-mono uppercase tracking-wider"
                style={{ color: preset.accent }}
              >
                diagnosis
              </span>
              <span className="text-[11.5px] text-dark-text-primary font-sans leading-snug">
                {preset.diagnosis}
              </span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-[10px] font-mono uppercase tracking-wider text-term-green">
                prescription
              </span>
              <span className="text-[11.5px] text-dark-text-secondary font-sans leading-snug">
                {preset.prescription}
              </span>
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
