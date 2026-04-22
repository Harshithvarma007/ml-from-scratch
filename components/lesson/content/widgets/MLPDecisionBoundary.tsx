'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Button, Readout } from './WidgetFrame'
import { Play, Pause, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'

// A live-training 2 → H → 1 MLP on either "moons" or "circles". Decision
// boundary updates in real time as the network learns. Width knob lets the
// user see that 3 neurons can't solve concentric circles, 8 can, 32 is
// excessive.

type DatasetName = 'moons' | 'circles' | 'xor'

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

function makeDataset(name: DatasetName, n = 160): Array<{ x: number; y: number; label: 0 | 1 }> {
  const rng = mulberry32(7)
  const out: Array<{ x: number; y: number; label: 0 | 1 }> = []
  if (name === 'moons') {
    for (let i = 0; i < n; i++) {
      const t = (i / n) * Math.PI
      const nx = gauss(rng) * 0.08
      const ny = gauss(rng) * 0.08
      if (i % 2 === 0) {
        out.push({ x: Math.cos(t) - 0.5 + nx, y: Math.sin(t) + ny, label: 0 })
      } else {
        out.push({ x: Math.cos(t) + 0.5 + nx, y: -Math.sin(t) + 0.3 + ny, label: 1 })
      }
    }
  } else if (name === 'circles') {
    for (let i = 0; i < n; i++) {
      const t = rng() * Math.PI * 2
      const r = i % 2 === 0 ? 0.5 : 1.1
      const nx = gauss(rng) * 0.05
      const ny = gauss(rng) * 0.05
      out.push({ x: r * Math.cos(t) + nx, y: r * Math.sin(t) + ny, label: (i % 2) as 0 | 1 })
    }
  } else {
    // noisy XOR
    for (let i = 0; i < n; i++) {
      const cx = i % 2 === 0 ? -0.6 : 0.6
      const cy = (i % 4 < 2 ? -0.6 : 0.6)
      const label = (((i % 2 === 0) !== (i % 4 < 2)) ? 1 : 0) as 0 | 1
      out.push({ x: cx + gauss(rng) * 0.3, y: cy + gauss(rng) * 0.3, label })
    }
  }
  return out
}

// Tiny MLP: 2 → H → 1, ReLU hidden, sigmoid output.
interface MLP {
  W1: number[][] // (H, 2)
  b1: number[] // (H,)
  W2: number[] // (H,)
  b2: number
}

function initMLP(H: number, rng: () => number): MLP {
  const W1: number[][] = []
  for (let i = 0; i < H; i++) {
    W1.push([gauss(rng) * Math.sqrt(2 / 2), gauss(rng) * Math.sqrt(2 / 2)])
  }
  const b1 = new Array(H).fill(0)
  const W2 = new Array(H).fill(0).map(() => gauss(rng) * Math.sqrt(2 / H))
  return { W1, b1, W2, b2: 0 }
}

function relu(z: number): number {
  return Math.max(0, z)
}
function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z))
}

function forwardBatch(net: MLP, data: Array<{ x: number; y: number }>) {
  // Returns { z1: (N,H), a1: (N,H), yhat: (N,) }
  const N = data.length
  const H = net.W1.length
  const z1: number[][] = []
  const a1: number[][] = []
  const yhat: number[] = []
  for (let n = 0; n < N; n++) {
    const row: number[] = []
    const act: number[] = []
    for (let h = 0; h < H; h++) {
      const z = net.W1[h][0] * data[n].x + net.W1[h][1] * data[n].y + net.b1[h]
      row.push(z)
      act.push(relu(z))
    }
    z1.push(row)
    a1.push(act)
    let z2 = net.b2
    for (let h = 0; h < H; h++) z2 += net.W2[h] * act[h]
    yhat.push(sigmoid(z2))
  }
  return { z1, a1, yhat }
}

function trainStep(
  net: MLP,
  data: Array<{ x: number; y: number; label: 0 | 1 }>,
  lr: number,
): number {
  const H = net.W1.length
  const { z1, a1, yhat } = forwardBatch(net, data)
  const N = data.length

  let loss = 0
  // Accumulate gradients
  const dW1 = net.W1.map((r) => r.map(() => 0))
  const db1 = new Array(H).fill(0)
  const dW2 = new Array(H).fill(0)
  let db2 = 0

  for (let n = 0; n < N; n++) {
    const y = data[n].label
    const p = yhat[n]
    // binary cross-entropy
    loss += -(y * Math.log(Math.max(p, 1e-9)) + (1 - y) * Math.log(Math.max(1 - p, 1e-9)))
    // δ at output layer (pre-sigmoid) collapses to p - y thanks to the BCE + sigmoid combo
    const delta2 = (p - y) / N
    db2 += delta2
    for (let h = 0; h < H; h++) {
      dW2[h] += delta2 * a1[n][h]
      // δ at hidden
      const d1h = net.W2[h] * delta2 * (z1[n][h] > 0 ? 1 : 0)
      db1[h] += d1h
      dW1[h][0] += d1h * data[n].x
      dW1[h][1] += d1h * data[n].y
    }
  }

  for (let h = 0; h < H; h++) {
    net.W1[h][0] -= lr * dW1[h][0]
    net.W1[h][1] -= lr * dW1[h][1]
    net.b1[h] -= lr * db1[h]
    net.W2[h] -= lr * dW2[h]
  }
  net.b2 -= lr * db2

  return loss / N
}

export default function MLPDecisionBoundary() {
  const [dataset, setDataset] = useState<DatasetName>('moons')
  const [hidden, setHidden] = useState(8)
  const [lr, setLr] = useState(0.3)
  const [playing, setPlaying] = useState(false)
  const [step, setStep] = useState(0)
  const [loss, setLoss] = useState(0.7)
  const netRef = useRef<MLP | null>(null)
  const rafRef = useRef<number | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)
  const [_, forceDraw] = useState(0)

  const data = useMemo(() => makeDataset(dataset), [dataset])

  const resetNet = () => {
    const rng = mulberry32(Date.now() & 0xffff)
    netRef.current = initMLP(hidden, rng)
    setStep(0)
    setLoss(0.7)
    forceDraw((n) => n + 1)
  }

  useEffect(() => {
    resetNet()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hidden, dataset])

  // Training loop
  useEffect(() => {
    if (!playing || !netRef.current) return
    const tick = () => {
      for (let i = 0; i < 3; i++) {
        if (!netRef.current) break
        const L = trainStep(netRef.current, data, lr)
        setLoss(L)
        setStep((s) => s + 1)
      }
      forceDraw((n) => n + 1)
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, lr])

  // Render
  useEffect(() => {
    const canvas = canvasRef.current
    const box = boxRef.current
    if (!canvas || !box || !netRef.current) return
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

      const pad = 20
      const plotW = w - pad * 2
      const plotH = h - pad * 2
      const X_MIN = -1.8
      const X_MAX = 1.8
      const toSx = (xv: number) => pad + ((xv - X_MIN) / (X_MAX - X_MIN)) * plotW
      const toSy = (yv: number) => pad + plotH - ((yv - X_MIN) / (X_MAX - X_MIN)) * plotH

      // Heatmap of decision surface
      const STEP = 6
      const imgData = ctx.createImageData(plotW, plotH)
      for (let py = 0; py < plotH; py += STEP) {
        for (let px = 0; px < plotW; px += STEP) {
          const xv = X_MIN + (px / plotW) * (X_MAX - X_MIN)
          const yv = X_MIN + (1 - py / plotH) * (X_MAX - X_MIN)
          const { yhat } = forwardBatch(netRef.current!, [{ x: xv, y: yv }])
          const q = yhat[0]
          const r = q < 0.5 ? 244 : 103
          const g = q < 0.5 ? 114 : 232
          const b8 = q < 0.5 ? 182 : 249
          const alpha = Math.abs(q - 0.5) * 2 * 80 + 20
          for (let dy = 0; dy < STEP && py + dy < plotH; dy++) {
            for (let dx = 0; dx < STEP && px + dx < plotW; dx++) {
              const idx = ((py + dy) * plotW + (px + dx)) * 4
              imgData.data[idx] = r
              imgData.data[idx + 1] = g
              imgData.data[idx + 2] = b8
              imgData.data[idx + 3] = alpha
            }
          }
        }
      }
      ctx.putImageData(imgData, pad, pad)

      // Data
      data.forEach((p) => {
        ctx.fillStyle = p.label === 1 ? '#67e8f9' : '#f472b6'
        ctx.strokeStyle = '#0a0a0a'
        ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.arc(toSx(p.x), toSy(p.y), 4, 0, Math.PI * 2)
        ctx.fill()
        ctx.stroke()
      })
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [_, data])

  const accuracy = useMemo(() => {
    if (!netRef.current) return 0
    const { yhat } = forwardBatch(netRef.current, data)
    let correct = 0
    for (let i = 0; i < data.length; i++) {
      const pred = yhat[i] > 0.5 ? 1 : 0
      if (pred === data[i].label) correct++
    }
    return correct / data.length
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [_, data])

  return (
    <WidgetFrame
      widgetName="MLPDecisionBoundary"
      label="live MLP — watch the decision boundary form"
      right={
        <>
          <span className="font-mono">2 → {hidden} (ReLU) → 1 (sigmoid)</span>
          <span className="text-dark-text-disabled">·</span>
          <span>BCE + SGD</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            {(['moons', 'circles', 'xor'] as DatasetName[]).map((d) => (
              <button
                key={d}
                onClick={() => setDataset(d)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono uppercase transition-all',
                  dataset === d
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
                )}
              >
                {d}
              </button>
            ))}
          </div>
          <Slider
            label="hidden H"
            value={hidden}
            min={1}
            max={32}
            step={1}
            onChange={(v) => setHidden(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-purple"
          />
          <Slider
            label="lr α"
            value={lr}
            min={0.01}
            max={1.5}
            step={0.01}
            onChange={setLr}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-2">
            <Button onClick={() => setPlaying((p) => !p)} variant="primary">
              {playing ? (
                <>
                  <Pause className="w-3 h-3 inline -mt-px mr-1" /> pause
                </>
              ) : (
                <>
                  <Play className="w-3 h-3 inline -mt-px mr-1" /> train
                </>
              )}
            </Button>
            <Button
              onClick={() => {
                setPlaying(false)
                resetNet()
              }}
            >
              <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
            </Button>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="step" value={String(step)} />
            <Readout label="loss" value={loss.toFixed(3)} accent="text-term-amber" />
            <Readout
              label="acc"
              value={`${(accuracy * 100).toFixed(0)}%`}
              accent={accuracy > 0.95 ? 'text-term-green' : 'text-term-amber'}
            />
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
