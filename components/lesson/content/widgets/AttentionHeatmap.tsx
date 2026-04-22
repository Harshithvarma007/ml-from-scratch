'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Causal attention heatmap on an 8-token sentence. We start from a hand-designed
// pre-softmax score matrix (so the pattern is readable: "the cat sat on the mat"
// tokens lean hard on "cat" and "mat"), then apply causal masking + temperature
// + softmax. Hovering a row shows which tokens that query attends to via a bar
// chart below the grid.

const TOKENS = ['the', 'cat', 'sat', 'on', 'the', 'mat', 'yesterday', 'afternoon']
const N = TOKENS.length

// Hand-designed "base scores" — these are the unscaled logits before softmax.
// Row i is query i. We seed it with a local-recency prior and a "content"
// prior that spikes on "cat" (idx=1) and "mat" (idx=5) for later tokens.
function buildBaseScores(): number[][] {
  const S: number[][] = []
  for (let i = 0; i < N; i++) {
    const row: number[] = []
    for (let j = 0; j < N; j++) {
      const distance = Math.abs(i - j)
      let s = -distance * 0.3 // recency prior
      // content spikes
      if (j === 1 && i >= 2) s += 1.6 // "cat" important after it appears
      if (j === 5 && i >= 5) s += 1.9 // "mat" even more important after it appears
      if (j === i) s += 0.4 // mild self-attention
      row.push(s)
    }
    S.push(row)
  }
  return S
}

const BASE = buildBaseScores()

function softmaxRow(row: number[]): number[] {
  const m = Math.max(...row)
  const e = row.map((v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

export default function AttentionHeatmap() {
  const [temperature, setTemperature] = useState(1.0)
  const [causal, setCausal] = useState(true)
  const [hover, setHover] = useState<number | null>(null)

  const attn = useMemo(() => {
    return BASE.map((row, i) => {
      const scaled = row.map((v, j) => {
        if (causal && j > i) return -1e9
        return v / Math.max(temperature, 0.01)
      })
      return softmaxRow(scaled)
    })
  }, [temperature, causal])

  const maxVal = Math.max(...attn.flat())
  const selectedRow = hover !== null ? attn[hover] : null

  return (
    <WidgetFrame
      widgetName="AttentionHeatmap"
      label="attention heatmap — hover rows to inspect a query"
      right={<span className="font-mono">&quot;{TOKENS.join(' ')}&quot;</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="temp"
            value={temperature}
            min={0.1}
            max={3.0}
            step={0.05}
            onChange={setTemperature}
            format={(v) => v.toFixed(2)}
            accent="accent-term-amber"
          />
          <button
            onClick={() => setCausal(!causal)}
            className={cn(
              'px-2.5 py-1 rounded text-[11px] font-mono uppercase transition-all',
              causal
                ? 'bg-dark-accent text-white'
                : 'border border-dark-border text-dark-text-secondary',
            )}
          >
            causal mask {causal ? 'on' : 'off'}
          </button>
          <div className="ml-auto flex items-center gap-4">
            <Readout
              label="query"
              value={hover !== null ? `${hover} (${TOKENS[hover]})` : '—'}
              accent="text-term-cyan"
            />
            <Readout label="max α" value={maxVal.toFixed(3)} accent="text-term-amber" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-cols-1 md:grid-cols-[1fr_200px] gap-3">
        <div className="flex flex-col min-h-0 gap-2">
          {/* Column labels */}
          <div className="flex pl-[90px]">
            {TOKENS.map((t, j) => (
              <div
                key={j}
                className="flex-1 text-center font-mono text-[10px] text-dark-text-muted truncate"
              >
                {t}
              </div>
            ))}
          </div>
          {/* Heatmap */}
          <div className="flex-1 flex flex-col gap-[2px] min-h-0">
            {attn.map((row, i) => (
              <div
                key={i}
                className={cn(
                  'flex items-center gap-1 cursor-pointer transition-all flex-1',
                  hover === i && 'ring-1 ring-term-cyan/70 rounded-sm',
                )}
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
              >
                <span className="w-[80px] pr-2 text-right font-mono text-[10px] text-dark-text-secondary truncate">
                  {TOKENS[i]}
                </span>
                <div className="flex-1 h-full flex gap-[2px]">
                  {row.map((v, j) => {
                    const isMasked = causal && j > i
                    return (
                      <div
                        key={j}
                        className="flex-1 rounded-[2px] flex items-center justify-center font-mono text-[9px] text-dark-text-primary tabular-nums min-h-0"
                        style={{
                          backgroundColor: isMasked
                            ? 'rgba(40, 40, 48, 0.5)'
                            : `rgba(167, 139, 250, ${0.1 + v * 0.85})`,
                        }}
                        title={`${TOKENS[i]} → ${TOKENS[j]}: ${v.toFixed(3)}`}
                      >
                        {!isMasked && v > 0.02 ? v.toFixed(2) : ''}
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
          <div className="text-[10px] font-mono text-dark-text-muted">
            {causal ? 'tokens can only see the past (and themselves).' : 'causal mask off — every token sees every other.'}
          </div>
        </div>

        {/* Side panel: bar chart for hovered row */}
        <div className="flex flex-col gap-2 min-w-0 overflow-hidden">
          <div className="text-[10px] font-mono uppercase tracking-wider text-term-cyan">
            {hover !== null ? `query: ${TOKENS[hover]}` : 'hover a row'}
          </div>
          <div className="flex-1 flex flex-col gap-1 min-h-0">
            {selectedRow ? (
              TOKENS.map((t, j) => {
                const v = selectedRow[j]
                const isMasked = causal && hover !== null && j > hover
                return (
                  <div key={j} className="flex items-center gap-1 font-mono text-[9.5px]">
                    <span className="w-[60px] text-right text-dark-text-secondary truncate">{t}</span>
                    <div className="flex-1 h-3 bg-dark-surface-elevated/40 rounded-sm overflow-hidden relative">
                      <div
                        className="absolute top-0 bottom-0 left-0"
                        style={{
                          width: `${v * 100}%`,
                          backgroundColor: isMasked ? '#4b5563' : 'rgba(167, 139, 250, 0.75)',
                        }}
                      />
                    </div>
                    <span className="w-8 text-right text-dark-text-primary tabular-nums">
                      {isMasked ? '—' : v.toFixed(2)}
                    </span>
                  </div>
                )
              })
            ) : (
              <div className="text-[10px] font-mono text-dark-text-disabled italic">
                hovering a row shows its attention distribution.
              </div>
            )}
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
