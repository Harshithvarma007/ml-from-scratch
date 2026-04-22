'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Side-by-side: the textbook formula vs the stable one. Crank the offset
// slider and the naive column turns to NaN. The stable column keeps working.
// Same inputs, same answer (when both survive), but one of them blows up
// past around z=710 because Math.exp(710) is Infinity.

const BASE = [2.0, 1.2, 0.6, -0.3, -1.1]

function naiveSoftmax(z: number[]): number[] {
  const e = z.map((v) => Math.exp(v))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

function stableSoftmax(z: number[]): number[] {
  const m = Math.max(...z)
  const e = z.map((v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

function fmtProb(v: number): string {
  if (!isFinite(v) || Number.isNaN(v)) return 'NaN'
  return (v * 100).toFixed(2) + '%'
}

function fmtExp(v: number): string {
  if (!isFinite(v)) return 'Inf'
  if (Number.isNaN(v)) return 'NaN'
  if (v === 0) return '0'
  if (v > 1e4) return v.toExponential(2)
  return v.toFixed(3)
}

export default function SoftmaxStability() {
  const [offset, setOffset] = useState(0)
  const logits = useMemo(() => BASE.map((v) => v + offset), [offset])
  const naiveExps = useMemo(() => logits.map((v) => Math.exp(v)), [logits])
  const naive = useMemo(() => naiveSoftmax(logits), [logits])
  const stable = useMemo(() => stableSoftmax(logits), [logits])

  const anyNaN = naive.some((v) => !isFinite(v) || Number.isNaN(v))

  return (
    <WidgetFrame
      widgetName="SoftmaxStability"
      label="numerical stability — naive vs stable softmax"
      right={<span className="font-mono">the &apos;subtract max&apos; trick</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="logit offset"
            value={offset}
            min={0}
            max={1000}
            step={5}
            onChange={setOffset}
            accent="accent-term-rose"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="naive"
              value={anyNaN ? 'broken' : 'ok'}
              accent={anyNaN ? 'text-term-rose' : 'text-term-green'}
            />
            <Readout label="stable" value="ok" accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 grid grid-cols-2 gap-2 p-3">
        {/* Naive column */}
        <div className="border border-dark-border rounded-md overflow-hidden flex flex-col bg-dark-bg">
          <div className="px-3 py-1.5 border-b border-dark-border bg-dark-surface-elevated/40 flex items-center justify-between">
            <span className="text-[11px] font-mono uppercase tracking-wider text-term-rose">
              naive
            </span>
            <span className="text-[10px] font-mono text-dark-text-disabled">
              softmax(z) = exp(z) / Σ exp(z)
            </span>
          </div>
          <div className="flex-1 p-3 font-mono text-[11px] overflow-auto">
            <table className="w-full">
              <thead className="text-dark-text-disabled uppercase text-[10px]">
                <tr>
                  <th className="text-left pb-1">z</th>
                  <th className="text-right pb-1">exp(z)</th>
                  <th className="text-right pb-1">p</th>
                </tr>
              </thead>
              <tbody>
                {logits.map((z, i) => {
                  const e = naiveExps[i]
                  const p = naive[i]
                  const broken = !isFinite(e) || Number.isNaN(p)
                  return (
                    <tr
                      key={i}
                      className={cn(
                        broken ? 'text-term-rose' : 'text-dark-text-primary'
                      )}
                    >
                      <td className="py-0.5 tabular-nums">{z.toFixed(1)}</td>
                      <td className="py-0.5 text-right tabular-nums">{fmtExp(e)}</td>
                      <td className="py-0.5 text-right tabular-nums">{fmtProb(p)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            {anyNaN && (
              <div className="mt-3 text-term-rose text-[10.5px]">
                Math.exp({logits[0].toFixed(0)}) = Inf · Inf / Inf = NaN ·
                model is cooked
              </div>
            )}
          </div>
        </div>

        {/* Stable column */}
        <div className="border border-dark-border rounded-md overflow-hidden flex flex-col bg-dark-bg">
          <div className="px-3 py-1.5 border-b border-dark-border bg-dark-surface-elevated/40 flex items-center justify-between">
            <span className="text-[11px] font-mono uppercase tracking-wider text-term-green">
              stable
            </span>
            <span className="text-[10px] font-mono text-dark-text-disabled">
              exp(z − max(z)) / Σ exp(z − max(z))
            </span>
          </div>
          <div className="flex-1 p-3 font-mono text-[11px] overflow-auto">
            <table className="w-full">
              <thead className="text-dark-text-disabled uppercase text-[10px]">
                <tr>
                  <th className="text-left pb-1">z − max</th>
                  <th className="text-right pb-1">exp</th>
                  <th className="text-right pb-1">p</th>
                </tr>
              </thead>
              <tbody>
                {logits.map((z, i) => {
                  const m = Math.max(...logits)
                  const shifted = z - m
                  const e = Math.exp(shifted)
                  const p = stable[i]
                  return (
                    <tr key={i} className="text-dark-text-primary">
                      <td className="py-0.5 tabular-nums">{shifted.toFixed(1)}</td>
                      <td className="py-0.5 text-right tabular-nums">{fmtExp(e)}</td>
                      <td className="py-0.5 text-right tabular-nums">{fmtProb(p)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            <div className="mt-3 text-term-green text-[10.5px]">
              max shift is algebraically free · all exponents ≤ 0 · numerically bulletproof
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
