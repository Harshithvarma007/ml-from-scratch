'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'

// Chinchilla-flavored scaling law plot. We use the canonical
// L(N, D) = E + A/N^α + B/D^β approximation with Hoffmann et al. (2022)
// parameters (A=406.4, B=410.7, E=1.69, α=0.34, β=0.28). For a given compute
// budget C (FLOPs), C ≈ 6 · N · D sweeps out a family of (N, D) pairs; the
// Chinchilla-optimal point is where the two terms contribute equally, which
// gives N_opt ∝ C^0.46 and D_opt ∝ C^0.54. We plot loss vs compute on log-log
// for four fixed model sizes, then overlay the optimal curve and a draggable
// compute marker.

const A = 406.4
const B_ = 410.7
const E = 1.69
const ALPHA = 0.34
const BETA = 0.28

function lossFromND(N: number, D: number): number {
  return E + A / Math.pow(N, ALPHA) + B_ / Math.pow(D, BETA)
}

function flops(N: number, D: number): number {
  return 6 * N * D
}

// Chinchilla-optimal for compute C: N_opt, D_opt such that both terms equal
function optimalAt(C: number): { N: number; D: number; loss: number } {
  // From the equal-contribution condition: A·α/N^α = B·β/D^β and 6ND = C.
  // Numerically solve over a log grid of N — cheap and correct.
  let best = { N: 1, D: 1, loss: Infinity }
  for (let i = 0; i < 400; i++) {
    const logN = 6 + (i / 399) * 8 // 1e6 to 1e14
    const N = Math.pow(10, logN)
    const D = C / (6 * N)
    if (D < 1e6 || D > 1e14) continue
    const l = lossFromND(N, D)
    if (l < best.loss) best = { N, D, loss: l }
  }
  return best
}

const MODEL_SIZES = [
  { name: 'small', N: 125e6, color: '#67e8f9' },
  { name: 'medium', N: 1.3e9, color: '#a78bfa' },
  { name: 'large', N: 13e9, color: '#fbbf24' },
  { name: 'huge', N: 70e9, color: '#f472b6' },
]

function formatFlops(C: number): string {
  if (C >= 1e24) return `${(C / 1e24).toFixed(1)}e24`
  if (C >= 1e21) return `${(C / 1e21).toFixed(1)}e21`
  if (C >= 1e18) return `${(C / 1e18).toFixed(1)}e18`
  return C.toExponential(1)
}

function formatParams(N: number): string {
  if (N >= 1e9) return `${(N / 1e9).toFixed(2)}B`
  if (N >= 1e6) return `${(N / 1e6).toFixed(1)}M`
  return N.toExponential(1)
}

function formatTokens(D: number): string {
  if (D >= 1e12) return `${(D / 1e12).toFixed(2)}T`
  if (D >= 1e9) return `${(D / 1e9).toFixed(1)}B`
  return D.toExponential(1)
}

export default function ScalingLawExplorer() {
  const [logC, setLogC] = useState(22) // default 1e22 FLOPs
  const C = Math.pow(10, logC)
  const opt = useMemo(() => optimalAt(C), [C])

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

      const padL = 54
      const padR = 16
      const padT = 16
      const padB = 34
      const plotW = w - padL - padR
      const plotH = h - padT - padB

      // Axis ranges — log-log
      const xMin = 18, xMax = 25 // log10(C)
      const yMin = 1.8, yMax = 4.0 // loss
      const toSx = (lc: number) => padL + ((lc - xMin) / (xMax - xMin)) * plotW
      const toSy = (l: number) => padT + plotH - ((l - yMin) / (yMax - yMin)) * plotH

      ctx.font = '10px "JetBrains Mono", monospace'
      ctx.strokeStyle = '#1e1e1e'
      ctx.lineWidth = 1
      ctx.fillStyle = '#555'
      // x grid
      ctx.textAlign = 'center'
      for (let lc = xMin; lc <= xMax; lc++) {
        const sx = toSx(lc)
        ctx.beginPath()
        ctx.moveTo(sx, padT)
        ctx.lineTo(sx, padT + plotH)
        ctx.stroke()
        ctx.fillText(`1e${lc}`, sx, padT + plotH + 14)
      }
      ctx.fillStyle = '#777'
      ctx.fillText('compute (FLOPs, log scale)', padL + plotW / 2, padT + plotH + 26)
      // y grid
      ctx.textAlign = 'right'
      ctx.fillStyle = '#555'
      for (let y = 2; y <= 4; y += 0.5) {
        const sy = toSy(y)
        ctx.beginPath()
        ctx.moveTo(padL, sy)
        ctx.lineTo(padL + plotW, sy)
        ctx.stroke()
        ctx.fillText(y.toFixed(1), padL - 6, sy + 3)
      }

      // y label
      ctx.save()
      ctx.translate(16, padT + plotH / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.textAlign = 'center'
      ctx.fillStyle = '#777'
      ctx.fillText('loss', 0, 0)
      ctx.restore()

      // Draw iso-N curves (fixed model size, vary D therefore C)
      MODEL_SIZES.forEach((m) => {
        ctx.strokeStyle = m.color
        ctx.globalAlpha = 0.7
        ctx.lineWidth = 1.6
        ctx.beginPath()
        let first = true
        for (let i = 0; i <= 200; i++) {
          const lc = xMin + (i / 200) * (xMax - xMin)
          const Cs = Math.pow(10, lc)
          const D = Cs / (6 * m.N)
          if (D < 1e6) continue
          const l = lossFromND(m.N, D)
          if (l < yMin || l > yMax) {
            first = true
            continue
          }
          const sx = toSx(lc)
          const sy = toSy(l)
          if (first) {
            ctx.moveTo(sx, sy)
            first = false
          } else ctx.lineTo(sx, sy)
        }
        ctx.stroke()
      })
      ctx.globalAlpha = 1

      // Draw the optimal envelope
      ctx.strokeStyle = '#f472b6'
      ctx.lineWidth = 2.2
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      let first = true
      for (let i = 0; i <= 80; i++) {
        const lc = xMin + (i / 80) * (xMax - xMin)
        const Cs = Math.pow(10, lc)
        const o = optimalAt(Cs)
        const sx = toSx(lc)
        const sy = toSy(o.loss)
        if (first) {
          ctx.moveTo(sx, sy)
          first = false
        } else ctx.lineTo(sx, sy)
      }
      ctx.stroke()
      ctx.setLineDash([])

      // Cursor compute marker
      const sxC = toSx(logC)
      ctx.strokeStyle = 'rgba(255,255,255,0.4)'
      ctx.setLineDash([3, 4])
      ctx.beginPath()
      ctx.moveTo(sxC, padT)
      ctx.lineTo(sxC, padT + plotH)
      ctx.stroke()
      ctx.setLineDash([])
      const syO = toSy(opt.loss)
      ctx.fillStyle = '#f472b6'
      ctx.beginPath()
      ctx.arc(sxC, syO, 5.5, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = '#0f0f1a'
      ctx.beginPath()
      ctx.arc(sxC, syO, 2.5, 0, Math.PI * 2)
      ctx.fill()

      // Legend
      ctx.font = '10px "JetBrains Mono", monospace'
      let lx = padL + 10
      const ly = padT + 14
      ctx.textAlign = 'left'
      MODEL_SIZES.forEach((m) => {
        ctx.strokeStyle = m.color
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(lx, ly - 4)
        ctx.lineTo(lx + 14, ly - 4)
        ctx.stroke()
        ctx.fillStyle = '#ccc'
        const text = `${m.name} · ${formatParams(m.N)}`
        ctx.fillText(text, lx + 18, ly)
        lx += 20 + ctx.measureText(text).width + 14
      })
      ctx.strokeStyle = '#f472b6'
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(lx, ly - 4)
      ctx.lineTo(lx + 14, ly - 4)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = '#f472b6'
      ctx.fillText('Chinchilla-optimal', lx + 18, ly)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(box)
    return () => ro.disconnect()
  }, [logC, opt])

  return (
    <WidgetFrame
      widgetName="ScalingLawExplorer"
      label="Chinchilla scaling law — loss vs compute"
      right={<span className="font-mono">L(N,D) = {E.toFixed(2)} + {A.toFixed(1)}/N^{ALPHA.toFixed(2)} + {B_.toFixed(1)}/D^{BETA.toFixed(2)}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="log10(C)"
            value={logC}
            min={18}
            max={25}
            step={0.1}
            onChange={setLogC}
            format={(v) => `1e${v.toFixed(1)}`}
            accent="accent-term-pink"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="C" value={`${formatFlops(C)} FLOPs`} />
            <Readout label="N*" value={formatParams(opt.N)} accent="text-term-pink" />
            <Readout label="D*" value={formatTokens(opt.D)} accent="text-term-pink" />
            <Readout label="loss*" value={opt.loss.toFixed(3)} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[1fr_250px] gap-4 overflow-hidden">
        <div ref={boxRef} className="relative min-h-0">
          <canvas ref={canvasRef} className="w-full h-full block" />
        </div>

        <div className="flex flex-col gap-3 min-w-0 overflow-auto">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            Chinchilla-optimal split
          </div>
          <div className="rounded border border-dark-border bg-dark-surface-elevated/40 p-3 font-mono text-[10.5px] leading-relaxed text-dark-text-muted">
            <div className="flex justify-between">
              <span>C</span>
              <span className="text-term-pink">{formatFlops(C)} FLOPs</span>
            </div>
            <div className="flex justify-between">
              <span>N*</span>
              <span className="text-term-pink tabular-nums">{formatParams(opt.N)} params</span>
            </div>
            <div className="flex justify-between">
              <span>D*</span>
              <span className="text-term-pink tabular-nums">{formatTokens(opt.D)} tokens</span>
            </div>
            <div className="flex justify-between">
              <span>D*/N*</span>
              <span className="text-term-amber tabular-nums">{(opt.D / opt.N).toFixed(1)} tok/param</span>
            </div>
            <div className="flex justify-between">
              <span>loss*</span>
              <span className="text-term-amber tabular-nums">{opt.loss.toFixed(3)}</span>
            </div>
          </div>

          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-1">
            named anchors
          </div>
          <div className="font-mono text-[10px] leading-relaxed text-dark-text-muted">
            <AnchorRow name="GPT-3 175B" C={3.14e23} active={Math.abs(logC - Math.log10(3.14e23)) < 0.1} onClick={() => setLogC(Math.log10(3.14e23))} />
            <AnchorRow name="Chinchilla 70B" C={5.76e23} active={Math.abs(logC - Math.log10(5.76e23)) < 0.1} onClick={() => setLogC(Math.log10(5.76e23))} />
            <AnchorRow name="Llama-2 7B" C={8.4e22} active={Math.abs(logC - Math.log10(8.4e22)) < 0.1} onClick={() => setLogC(Math.log10(8.4e22))} />
            <AnchorRow name="research toy" C={1e20} active={Math.abs(logC - 20) < 0.1} onClick={() => setLogC(20)} />
          </div>

          <div className="mt-1 font-mono text-[10.5px] leading-relaxed text-dark-text-muted border-t border-dark-border pt-2">
            rule of thumb: ~20 tokens per parameter. GPT-3 trained at ~1.7 tok/param — undertrained; Chinchilla redeemed it.
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function AnchorRow({ name, C, active, onClick }: { name: string; C: number; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`flex justify-between w-full px-2 py-0.5 rounded hover:bg-dark-surface-elevated/40 transition-colors text-left ${active ? 'text-term-amber' : 'text-dark-text-secondary'}`}
    >
      <span>{name}</span>
      <span className="tabular-nums">{formatFlops(C)}</span>
    </button>
  )
}
