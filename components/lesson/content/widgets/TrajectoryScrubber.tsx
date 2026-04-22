'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Shows the numeric trajectory of x_n and f(x_n) as the reader scrubs through
// iterations. Intentionally small and textual — paired with the graphical
// explorer above it, this one's job is to make the numbers feel inevitable.

const X0 = 5
const LR = 0.1
const STEPS = 25

function trace() {
  const rows: { n: number; x: number; loss: number }[] = []
  let x = X0
  for (let n = 0; n <= STEPS; n++) {
    rows.push({ n, x, loss: x * x })
    x = x - LR * 2 * x
  }
  return rows
}

export default function TrajectoryScrubber() {
  const rows = useMemo(() => trace(), [])
  const [cursor, setCursor] = useState(0)

  return (
    <WidgetFrame
      widgetName="TrajectoryScrubber"
      label="25 steps by hand"
      right={
        <>
          <span>x₀ = 5</span>
          <span className="text-dark-text-disabled">·</span>
          <span>α = 0.1</span>
        </>
      }
      aspect="wide"
      controls={
        <Slider
          label="step"
          value={cursor}
          min={0}
          max={STEPS}
          step={1}
          onChange={setCursor}
          format={(n) => String(n).padStart(2, ' ') + '/' + String(STEPS)}
          accent="accent-term-amber"
        />
      }
    >
      <div className="absolute inset-0 grid grid-cols-[1fr_1fr] gap-4 p-6">
        {/* Left: numeric table, dim before cursor, bright at cursor, hidden after */}
        <div className="overflow-y-auto pr-2">
          <table className="w-full font-mono text-[12px] tabular-nums">
            <thead className="text-dark-text-disabled uppercase tracking-wider text-[10px]">
              <tr>
                <th className="text-left py-1">step</th>
                <th className="text-right py-1">x</th>
                <th className="text-right py-1">f(x)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const active = r.n === cursor
                const past = r.n < cursor
                return (
                  <tr
                    key={r.n}
                    className={cn(
                      'transition-all',
                      active &&
                        'bg-dark-accent/10 text-dark-text-primary shadow-[inset_2px_0_0_#a78bfa]',
                      past && !active && 'text-dark-text-muted',
                      !past && !active && 'text-dark-text-disabled'
                    )}
                  >
                    <td className="py-0.5 pl-2 text-left">{String(r.n).padStart(2, ' ')}</td>
                    <td className="py-0.5 text-right pr-2">{r.x.toFixed(4)}</td>
                    <td
                      className={cn(
                        'py-0.5 text-right pr-2',
                        active && 'text-term-amber'
                      )}
                    >
                      {r.loss.toFixed(4)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* Right: big current readout + mini bar graph of loss decay */}
        <div className="flex flex-col justify-center gap-4">
          <div className="space-y-3">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-dark-text-disabled font-mono">
                step
              </div>
              <div className="font-mono text-[32px] text-dark-text-primary tabular-nums leading-none">
                {String(cursor).padStart(2, '0')}
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-dark-text-disabled font-mono">
                x
              </div>
              <div className="font-mono text-[20px] text-dark-text-primary tabular-nums leading-none">
                {rows[cursor].x.toFixed(5)}
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-dark-text-disabled font-mono">
                f(x) — loss
              </div>
              <div className="font-mono text-[20px] text-term-amber tabular-nums leading-none">
                {rows[cursor].loss.toFixed(5)}
              </div>
            </div>
          </div>

          {/* Mini bar graph — loss decay */}
          <div className="flex items-end gap-[2px] h-14 border-b border-dark-border">
            {rows.map((r, i) => {
              const h = Math.max(1, (r.loss / rows[0].loss) * 56)
              return (
                <div
                  key={i}
                  className={cn(
                    'flex-1 transition-all',
                    i <= cursor ? 'bg-term-amber/80' : 'bg-dark-border/60'
                  )}
                  style={{ height: `${h}px` }}
                />
              )
            })}
          </div>
          <div className="text-[10px] font-mono text-dark-text-disabled uppercase tracking-wider">
            loss decay
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
