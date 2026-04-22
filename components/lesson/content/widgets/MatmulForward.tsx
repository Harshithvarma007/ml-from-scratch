'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Walk the reader through ŷ = X @ w + b. Hover a row of X to highlight the
// single dot-product that produces that row's prediction. Crisper than a
// diagram — it IS the diagram, and it does the arithmetic.

interface MatmulState {
  X: number[][] // N x D
  w: number[] // D
  b: number
}

function initState(): MatmulState {
  // A hand-picked example — house price prediction with 3 features
  // (sqft (×1000), bedrooms, age) on 5 houses.
  const X = [
    [1.4, 2, 10],
    [2.1, 3, 5],
    [3.2, 4, 2],
    [1.8, 2, 20],
    [2.6, 3, 15],
  ]
  const w = [100, 25, -2]
  const b = 20
  return { X, w, b }
}

export default function MatmulForward() {
  const [state] = useState<MatmulState>(() => initState())
  const [hover, setHover] = useState<number | null>(0)

  const predictions = useMemo(() => {
    const { X, w, b } = state
    return X.map((row) => row.reduce((s, xi, i) => s + xi * w[i], 0) + b)
  }, [state])

  const { X, w, b } = state
  const featureLabels = ['sqft (×1k)', 'bedrooms', 'age (yr)']

  return (
    <WidgetFrame
      widgetName="MatmulForward"
      label="the forward pass — a matrix-vector product"
      right={<span className="font-mono">ŷ = X · w + b</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="text-[11px] font-mono text-dark-text-muted">
            hover a row to see which dot-product becomes ŷᵢ
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="shape" value={`(${X.length}×${w.length}) · (${w.length}) = (${X.length})`} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 font-mono text-[11px] overflow-auto flex items-center justify-center">
        <div className="flex items-center gap-5">
          {/* X matrix */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-dark-text-disabled mb-1.5 text-center">
              X  — inputs
            </div>
            <table className="border-collapse">
              <thead>
                <tr>
                  {featureLabels.map((f) => (
                    <th
                      key={f}
                      className="text-[9px] font-normal text-dark-text-disabled px-2 py-0.5"
                    >
                      {f}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {X.map((row, i) => (
                  <tr
                    key={i}
                    onMouseEnter={() => setHover(i)}
                    className={cn(
                      'cursor-pointer transition-colors',
                      hover === i ? 'bg-dark-accent/10' : 'hover:bg-white/[0.02]'
                    )}
                  >
                    {row.map((v, j) => (
                      <td
                        key={j}
                        className={cn(
                          'px-2 py-1 text-right tabular-nums border border-dark-border',
                          hover === i ? 'text-dark-text-primary' : 'text-dark-text-secondary'
                        )}
                      >
                        {v.toFixed(1)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="text-dark-accent text-lg">·</div>

          {/* w vector (vertical) */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-dark-text-disabled mb-1.5 text-center">
              w  — weights
            </div>
            <table className="border-collapse">
              <tbody>
                {w.map((v, i) => (
                  <tr key={i}>
                    <td
                      className={cn(
                        'px-3 py-1 text-right tabular-nums border border-dark-border font-semibold',
                        'text-term-purple'
                      )}
                    >
                      {v >= 0 ? `+${v}` : v}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="text-[9px] text-dark-text-disabled text-center mt-1">
              one per feature
            </div>
          </div>

          <div className="text-dark-accent text-lg">+</div>

          {/* b scalar */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-dark-text-disabled mb-1.5 text-center">
              b
            </div>
            <div className="px-3 py-1 text-right tabular-nums border border-dark-border text-term-amber">
              {b}
            </div>
          </div>

          <div className="text-dark-accent text-lg">=</div>

          {/* ŷ vector */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-dark-text-disabled mb-1.5 text-center">
              ŷ  — prices ($k)
            </div>
            <table className="border-collapse">
              <tbody>
                {predictions.map((p, i) => (
                  <tr
                    key={i}
                    onMouseEnter={() => setHover(i)}
                    className={cn(
                      'cursor-pointer transition-colors',
                      hover === i ? 'bg-term-amber/10' : 'hover:bg-white/[0.02]'
                    )}
                  >
                    <td
                      className={cn(
                        'px-3 py-1 text-right tabular-nums border border-dark-border',
                        hover === i ? 'text-term-amber font-semibold' : 'text-dark-text-secondary'
                      )}
                    >
                      {p.toFixed(1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Work shown below */}
      {hover !== null && (
        <div className="absolute bottom-3 left-5 right-5 p-3 rounded border border-dark-border bg-dark-surface/80 backdrop-blur-sm">
          <div className="text-[10px] uppercase tracking-wider text-dark-text-disabled mb-1">
            row {hover} · the dot-product that makes ŷ{sub(hover)}
          </div>
          <div className="font-mono text-[11.5px] tabular-nums text-dark-text-primary flex flex-wrap items-center gap-2">
            ŷ{sub(hover)} ={' '}
            {X[hover].map((xi, j) => (
              <span key={j}>
                <span className="text-dark-text-muted">{xi.toFixed(1)}</span>
                <span className="text-dark-accent mx-1">·</span>
                <span className="text-term-purple">{w[j]}</span>
                {j < X[hover].length - 1 && <span className="text-dark-text-disabled"> + </span>}
              </span>
            ))}{' '}
            <span className="text-dark-text-disabled">+</span>{' '}
            <span className="text-term-amber">{b}</span>{' '}
            <span className="text-dark-accent mx-1">=</span>{' '}
            <span className="text-term-amber font-semibold">
              {predictions[hover].toFixed(1)}
            </span>
          </div>
        </div>
      )}
    </WidgetFrame>
  )
}

function sub(n: number): string {
  return String(n)
    .split('')
    .map((d) => '₀₁₂₃₄₅₆₇₈₉'[Number(d)] ?? d)
    .join('')
}
