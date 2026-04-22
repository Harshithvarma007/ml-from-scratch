'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Show the variance of the batch-mean estimate as a function of batch size.
// With N=1 the batch mean IS the sample — estimate is worthless. With N=128
// it's a solid estimate of the true mean. Dramatize why BatchNorm needs large
// batches and why LayerNorm doesn't.

function stdError(N: number, trueStd: number = 1): number {
  // Standard error of the sample mean for a N-sample batch with known std.
  return trueStd / Math.sqrt(N)
}

export default function BatchSizeFailure() {
  const [batchSize, setBatchSize] = useState(32)

  // True distribution: N(0, 1). We show what ±1 standard-error bar on the
  // sample mean looks like for different batch sizes.
  const errorBar = useMemo(() => stdError(batchSize), [batchSize])

  const ranges = [1, 2, 4, 8, 16, 32, 64, 128, 256]

  return (
    <WidgetFrame
      widgetName="BatchSizeFailure"
      label="batch size matters — the batch-statistic variance"
      right={<span className="font-mono">standard error ≈ σ / √N</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="batch N"
            value={batchSize}
            min={1}
            max={256}
            step={1}
            onChange={(v) => setBatchSize(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="std. error"
              value={errorBar.toFixed(3)}
              accent={errorBar > 0.3 ? 'text-term-rose' : 'text-term-green'}
            />
            <Readout
              label="verdict"
              value={errorBar > 0.3 ? 'unreliable' : 'ok'}
              accent={errorBar > 0.3 ? 'text-term-rose' : 'text-term-green'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 overflow-auto">
        <div className="max-w-[880px] mx-auto">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-3">
            each row is a different batch size · bar shows the ±1 std-error range of the batch mean
          </div>
          <div className="space-y-2">
            {ranges.map((N) => {
              const err = stdError(N)
              const isCurrent = N === batchSize
              return (
                <div
                  key={N}
                  className={cn(
                    'grid grid-cols-[60px_1fr_80px] items-center gap-4 p-2 rounded',
                    isCurrent && 'bg-dark-accent/10 ring-1 ring-dark-accent/40'
                  )}
                >
                  <span className="font-mono text-[12px] text-dark-text-primary text-right">
                    N = {N}
                  </span>
                  <div className="relative h-6 bg-dark-surface-elevated/40 rounded">
                    {/* axis line at the "true mean" */}
                    <div className="absolute inset-y-0 left-1/2 w-px bg-dark-border" />
                    {/* error bar */}
                    <div
                      className={cn(
                        'absolute top-1/2 -translate-y-1/2 h-3 rounded',
                        err > 0.3 ? 'bg-term-rose/50' : 'bg-term-green/50'
                      )}
                      style={{
                        left: `${50 - Math.min(50, err * 50)}%`,
                        right: `${50 - Math.min(50, err * 50)}%`,
                      }}
                    />
                    {/* center dot */}
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-white" />
                  </div>
                  <span
                    className={cn(
                      'font-mono text-[11px] tabular-nums',
                      err > 0.3 ? 'text-term-rose' : 'text-term-green'
                    )}
                  >
                    ± {err.toFixed(3)}
                  </span>
                </div>
              )
            })}
          </div>

          <div className="mt-5 p-3 rounded border border-dark-border bg-dark-surface-elevated/30 text-[11.5px] font-sans text-dark-text-secondary leading-relaxed">
            At <span className="text-term-rose">N=1</span> the &ldquo;batch mean&rdquo; is
            just the single sample — there is no statistic to estimate, and BatchNorm
            degenerates to a no-op (or worse, adds noise). At{' '}
            <span className="text-term-green">N=128</span> the error bar is small enough that
            batch statistics are a good proxy for population statistics. This scaling is why
            BatchNorm used to be fine for 2015-era vision models (large batches of
            independent images) and falls apart for transformers (tiny per-device batches,
            sequences that share context).
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
