'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// A 10×10 confusion matrix for a "97.5% accurate" MNIST classifier. Numbers
// are hand-curated to look like a real result: diagonal dominates, with the
// classic confusion pairs (4↔9, 3↔8, 5↔3) highlighted. Click a cell to see
// the count + per-class metrics on the side. Pure static data — this is a
// read-and-click widget, not a live simulation.

// cell[actual][predicted] = count. 1000 examples per class.
const MATRIX: number[][] = [
  // 0
  [980, 0, 1, 1, 0, 3, 10, 1, 4, 0],
  // 1
  [0, 1124, 2, 2, 0, 1, 2, 2, 2, 0],
  // 2
  [4, 2, 1001, 6, 4, 0, 2, 7, 5, 1],
  // 3
  [1, 0, 5, 987, 0, 12, 0, 3, 2, 0],
  // 4
  [1, 1, 3, 0, 962, 0, 4, 2, 1, 8],
  // 5
  [3, 0, 1, 13, 1, 863, 5, 1, 4, 1],
  // 6
  [5, 2, 1, 0, 4, 6, 938, 0, 2, 0],
  // 7
  [1, 3, 9, 2, 1, 0, 0, 1005, 2, 5],
  // 8
  [4, 0, 3, 5, 2, 6, 4, 4, 944, 2],
  // 9
  [4, 3, 0, 5, 13, 3, 1, 5, 5, 970],
]

export default function ConfusionMatrixLive() {
  const [cell, setCell] = useState<[number, number] | null>([4, 9])

  const stats = useMemo(() => {
    let total = 0, correct = 0
    for (let i = 0; i < 10; i++) {
      for (let j = 0; j < 10; j++) {
        total += MATRIX[i][j]
        if (i === j) correct += MATRIX[i][j]
      }
    }
    const perClass: { precision: number; recall: number; support: number }[] = []
    for (let k = 0; k < 10; k++) {
      const tp = MATRIX[k][k]
      let fp = 0, fn = 0, support = 0
      for (let i = 0; i < 10; i++) {
        support += MATRIX[k][i]                          // row sum = support
        if (i !== k) fn += MATRIX[k][i]                  // row - diagonal = false negatives (for class k)
        if (i !== k) fp += MATRIX[i][k]                  // column - diagonal = false positives
      }
      perClass.push({
        precision: tp / (tp + fp),
        recall: tp / (tp + fn),
        support,
      })
    }
    return { accuracy: correct / total, total, perClass }
  }, [])

  const maxCell = Math.max(...MATRIX.flat())
  const offdiagMax = Math.max(
    ...MATRIX.flatMap((row, i) => row.map((v, j) => (i === j ? 0 : v))),
  )

  // Click-to-inspect panel content
  const info = cell ? MATRIX[cell[0]][cell[1]] : null
  const isDiag = cell ? cell[0] === cell[1] : false

  const topConfusions = useMemo(() => {
    const out: { actual: number; predicted: number; count: number }[] = []
    for (let i = 0; i < 10; i++) {
      for (let j = 0; j < 10; j++) {
        if (i !== j && MATRIX[i][j] > 0) out.push({ actual: i, predicted: j, count: MATRIX[i][j] })
      }
    }
    return out.sort((a, b) => b.count - a.count).slice(0, 5)
  }, [])

  return (
    <WidgetFrame
      widgetName="ConfusionMatrixLive"
      label="confusion matrix — where the classifier makes its mistakes"
      right={<span className="font-mono">97.6% test accuracy · 10,000 examples</span>}
      aspect="wide"
      controls={
        <div className="flex items-center gap-4">
          <div className="text-[11px] font-mono text-dark-text-muted">
            click a cell to see the count and interpretation
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="accuracy" value={`${(stats.accuracy * 100).toFixed(1)}%`} accent="text-term-green" />
            <Readout label="total" value={stats.total.toLocaleString()} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 grid grid-cols-1 md:grid-cols-[1fr_300px] gap-4 overflow-auto">
        {/* Matrix */}
        <div>
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-2 text-center">
            predicted →
          </div>
          <div className="flex items-start gap-1">
            <div className="flex flex-col gap-1 mr-1">
              <div className="w-5 h-5" />
              {Array.from({ length: 10 }).map((_, i) => (
                <div key={i} className="w-5 h-8 flex items-center justify-center text-[10px] font-mono text-dark-text-disabled">
                  {i}
                </div>
              ))}
              <div className="text-[10px] font-mono text-dark-text-disabled -rotate-90 mt-14 whitespace-nowrap">
                ← actual
              </div>
            </div>
            <div className="flex-1">
              <div className="grid grid-cols-10 gap-1 mb-1">
                {Array.from({ length: 10 }).map((_, j) => (
                  <div key={j} className="h-5 flex items-center justify-center text-[10px] font-mono text-dark-text-disabled">
                    {j}
                  </div>
                ))}
              </div>
              {MATRIX.map((row, i) => (
                <div key={i} className="grid grid-cols-10 gap-1 mb-1">
                  {row.map((v, j) => {
                    const ratio = i === j ? v / maxCell : v / Math.max(offdiagMax, 1)
                    const bg = i === j
                      ? `rgba(74, 222, 128, ${0.15 + ratio * 0.6})`
                      : `rgba(244, 114, 182, ${0.04 + ratio * 0.5})`
                    const selected = cell && cell[0] === i && cell[1] === j
                    return (
                      <button
                        key={j}
                        onClick={() => setCell([i, j])}
                        className={cn(
                          'h-8 rounded text-[9.5px] font-mono tabular-nums transition-all',
                          selected ? 'ring-2 ring-white' : 'hover:ring-1 hover:ring-dark-accent/60'
                        )}
                        style={{
                          backgroundColor: bg,
                          color: i === j ? '#4ade80' : v > 5 ? '#f472b6' : '#666',
                        }}
                      >
                        {v}
                      </button>
                    )
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Side panel */}
        <div className="flex flex-col gap-3 min-h-0 overflow-auto">
          {cell && (
            <div className="rounded border border-dark-border bg-dark-surface-elevated/30 p-3 font-mono text-[11.5px]">
              <div className="text-[10px] uppercase tracking-wider text-dark-text-disabled mb-1">
                actual = {cell[0]} · predicted = {cell[1]}
              </div>
              <div className="text-[14px] text-dark-text-primary font-semibold tabular-nums">
                {info} examples
              </div>
              <div className={cn('text-[11px] mt-1', isDiag ? 'text-term-green' : 'text-term-pink')}>
                {isDiag
                  ? 'correct prediction (diagonal)'
                  : `${cell[0]} mistaken for ${cell[1]}`}
              </div>
              <div className="text-[10.5px] text-dark-text-muted mt-2 leading-relaxed">
                {isDiag
                  ? `precision ${(stats.perClass[cell[0]].precision * 100).toFixed(1)}% · recall ${(stats.perClass[cell[0]].recall * 100).toFixed(1)}%`
                  : ''}
              </div>
            </div>
          )}

          <div className="rounded border border-dark-border bg-dark-bg p-3">
            <div className="text-[10px] uppercase tracking-wider text-dark-text-disabled mb-2">
              top 5 confusion pairs
            </div>
            <div className="space-y-1 font-mono text-[11px]">
              {topConfusions.map((c, i) => (
                <div key={i} className="flex items-center justify-between">
                  <span className="text-dark-text-muted">
                    {c.actual} → <span className="text-term-pink">{c.predicted}</span>
                  </span>
                  <span className="text-dark-text-primary tabular-nums">{c.count}</span>
                </div>
              ))}
            </div>
            <div className="text-[10px] text-dark-text-disabled mt-3 leading-relaxed">
              Classic ambiguities: 4/9 share the same vertical bar with a closed vs open top;
              3/5 share curly loops; 7/2 share a slanted top stroke.
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
