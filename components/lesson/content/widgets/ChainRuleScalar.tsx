'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider } from './WidgetFrame'
import { cn } from '@/lib/utils'

// A scalar composition: u = 3x + 2,  v = u²,  L = sin(v).  Four values, three
// local derivatives, two ways of computing the global one. Slide x, watch the
// forward values and the backward gradients update together.

function forward(x: number) {
  const u = 3 * x + 2
  const v = u * u
  const L = Math.sin(v)
  return { u, v, L }
}

export default function ChainRuleScalar() {
  const [x, setX] = useState(0.4)
  const { u, v, L } = useMemo(() => forward(x), [x])

  // Local derivatives
  const du_dx = 3 // d(3x+2)/dx
  const dv_du = 2 * u // d(u²)/du
  const dL_dv = Math.cos(v) // d(sin v)/dv

  // Chained: dL/dx = dL/dv · dv/du · du/dx
  const dL_dx = dL_dv * dv_du * du_dx

  return (
    <WidgetFrame
      widgetName="ChainRuleScalar"
      label="chain rule — a scalar walkthrough"
      right={<span className="font-mono">L = sin( (3x + 2)² )</span>}
      aspect="wide"
      controls={
        <Slider
          label="x"
          value={x}
          min={-2}
          max={2}
          step={0.01}
          onChange={setX}
          accent="accent-dark-accent"
        />
      }
    >
      <div className="absolute inset-0 p-6 overflow-auto">
        <div className="max-w-[880px] mx-auto grid grid-cols-4 gap-3 font-mono text-[12px]">
          {/* Headers */}
          <Header label="x" sub="input" />
          <Header label="u = 3x + 2" sub="affine" />
          <Header label="v = u²" sub="square" />
          <Header label="L = sin(v)" sub="loss" />

          {/* Forward values */}
          <ValueCell label="value" value={x.toFixed(4)} accent="text-term-cyan" />
          <ValueCell label="value" value={u.toFixed(4)} accent="text-term-cyan" />
          <ValueCell label="value" value={v.toFixed(4)} accent="text-term-cyan" />
          <ValueCell label="value" value={L.toFixed(4)} accent="text-term-amber" />

          {/* Local derivatives */}
          <ValueCell label="du/dx" value={String(du_dx)} accent="text-term-purple" />
          <ValueCell label="dv/du = 2u" value={dv_du.toFixed(4)} accent="text-term-purple" />
          <ValueCell label="dL/dv = cos v" value={dL_dv.toFixed(4)} accent="text-term-purple" />
          <div />

          {/* Incoming upstream gradient during backprop */}
          <ValueCell
            label="dL/dx"
            value={dL_dx.toFixed(4)}
            accent="text-term-rose"
            highlight
          />
          <ValueCell
            label="dL/du = dL/dv · dv/du"
            value={(dL_dv * dv_du).toFixed(4)}
            accent="text-term-rose"
            highlight
          />
          <ValueCell
            label="dL/dv"
            value={dL_dv.toFixed(4)}
            accent="text-term-rose"
            highlight
          />
          <ValueCell label="dL/dL" value="1.0000" accent="text-term-rose" highlight />
        </div>

        <div className="max-w-[880px] mx-auto mt-6 flex items-center gap-4 text-[11px] font-mono">
          <div className="flex items-center gap-2">
            <span className="inline-block w-3 h-3 rounded bg-term-cyan/60" />
            <span className="text-dark-text-muted">forward values (left → right)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-block w-3 h-3 rounded bg-term-purple/60" />
            <span className="text-dark-text-muted">local derivative at each node</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-block w-3 h-3 rounded bg-term-rose/60" />
            <span className="text-dark-text-muted">backward gradients (right → left)</span>
          </div>
        </div>

        <div className="max-w-[880px] mx-auto mt-4 p-3 rounded border border-dark-border bg-dark-surface-elevated/30 font-mono text-[11.5px]">
          <div className="text-dark-text-disabled uppercase tracking-wider text-[10px] mb-1">
            the rule, restated
          </div>
          <div className="text-dark-text-primary">
            dL/dx  =  dL/dv · dv/du · du/dx
          </div>
          <div className="text-dark-text-secondary">
            {'       '} = {dL_dv.toFixed(4)} · {dv_du.toFixed(4)} · {du_dx} ={' '}
            <span className="text-term-rose font-semibold">{dL_dx.toFixed(4)}</span>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function Header({ label, sub }: { label: string; sub: string }) {
  return (
    <div className="text-center border-b border-dark-border pb-1">
      <div className="text-dark-text-primary font-semibold">{label}</div>
      <div className="text-[10px] text-dark-text-disabled uppercase tracking-wider">{sub}</div>
    </div>
  )
}

function ValueCell({
  label,
  value,
  accent,
  highlight,
}: {
  label: string
  value: string
  accent?: string
  highlight?: boolean
}) {
  return (
    <div
      className={cn(
        'rounded p-2 border',
        highlight ? 'border-term-rose/40 bg-term-rose/[0.05]' : 'border-dark-border bg-dark-surface-elevated/30'
      )}
    >
      <div className="text-[10px] uppercase tracking-wider text-dark-text-disabled mb-0.5">
        {label}
      </div>
      <div className={cn('tabular-nums', accent ?? 'text-dark-text-primary')}>{value}</div>
    </div>
  )
}
