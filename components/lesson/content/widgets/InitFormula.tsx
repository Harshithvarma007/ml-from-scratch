'use client'

import { useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// A reference-card widget. Click a recipe, see its formula, the activations
// it's meant for, and a one-line PyTorch call. Non-interactive in the ML
// sense — a lookup table the reader can scan.

type Recipe = {
  name: string
  description: string
  formula: string
  variance: string
  bestFor: string
  pytorch: string
  color: string
}

const RECIPES: Recipe[] = [
  {
    name: 'Xavier (Glorot)',
    description:
      'Balances forward and backward variance. The recipe that made tanh networks trainable in 2010.',
    formula: 'W ~ 𝒩(0, 2 / (fan_in + fan_out))',
    variance: 'Var(W) = 2 / (fan_in + fan_out)',
    bestFor: 'tanh, sigmoid, linear activations',
    pytorch: 'nn.init.xavier_normal_(layer.weight)',
    color: '#60a5fa',
  },
  {
    name: 'He (Kaiming)',
    description:
      'Designed for ReLU — which kills half the activations, so we compensate by doubling the variance.',
    formula: 'W ~ 𝒩(0, 2 / fan_in)',
    variance: 'Var(W) = 2 / fan_in',
    bestFor: 'ReLU, Leaky ReLU, GELU',
    pytorch: "nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')",
    color: '#fbbf24',
  },
  {
    name: 'LeCun',
    description:
      'The original (1998) fan-in-only init. Still a perfect fit for SELU and a few niche activations.',
    formula: 'W ~ 𝒩(0, 1 / fan_in)',
    variance: 'Var(W) = 1 / fan_in',
    bestFor: 'SELU, and when matched to the activation',
    pytorch: 'nn.init.kaiming_normal_(layer.weight, nonlinearity=\'linear\')',
    color: '#a78bfa',
  },
  {
    name: 'Orthogonal',
    description:
      'Preserves variance exactly — the weight matrix is a rotation/reflection. Gold standard for RNNs and very deep residual nets.',
    formula: 'W = U from QR decomposition of a 𝒩(0, 1) matrix',
    variance: 'Var preserved exactly, not approximately',
    bestFor: 'RNNs, very deep nets, residual blocks',
    pytorch: 'nn.init.orthogonal_(layer.weight)',
    color: '#4ade80',
  },
]

export default function InitFormula() {
  const [active, setActive] = useState(1)
  const r = RECIPES[active]

  return (
    <WidgetFrame
      widgetName="InitFormula"
      label="the four initialisation recipes you&apos;ll actually use"
      right={<span className="font-mono">variance-preserving · Gaussian · layer-local</span>}
      aspect="wide"
      controls={
        <div className="flex items-center gap-4">
          <div className="text-[11px] font-mono text-dark-text-muted">
            click a card to see its formula + PyTorch one-liner
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="fan_in" value="input dim" />
            <Readout label="fan_out" value="output dim" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 grid grid-cols-1 md:grid-cols-[260px_1fr] gap-5">
        {/* Left: recipe list */}
        <div className="flex flex-col gap-2">
          {RECIPES.map((recipe, i) => (
            <button
              key={recipe.name}
              onClick={() => setActive(i)}
              className={cn(
                'text-left rounded border p-3 transition-all',
                active === i
                  ? 'border-dark-accent bg-dark-accent/[0.06]'
                  : 'border-dark-border hover:border-dark-border-hover'
              )}
              style={active === i ? { borderColor: recipe.color } : {}}
            >
              <div className="font-mono text-[12px] text-dark-text-primary font-semibold">
                {recipe.name}
              </div>
              <div className="font-mono text-[10.5px] text-dark-text-muted mt-0.5">
                {recipe.bestFor}
              </div>
            </button>
          ))}
        </div>

        {/* Right: details of active */}
        <div className="rounded border border-dark-border bg-dark-surface-elevated/30 p-4 overflow-auto">
          <div
            className="font-mono text-[10px] uppercase tracking-wider mb-1"
            style={{ color: r.color }}
          >
            {r.name}
          </div>
          <p className="font-sans text-[13px] text-dark-text-secondary leading-relaxed mb-4">
            {r.description}
          </p>

          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-1">
            formula
          </div>
          <div className="font-mono text-[13px] text-dark-text-primary mb-4 bg-dark-bg p-3 rounded border border-dark-border">
            {r.formula}
          </div>

          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-1">
            variance
          </div>
          <div className="font-mono text-[12px] text-dark-text-primary mb-4">
            {r.variance}
          </div>

          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-1">
            best paired with
          </div>
          <div className="font-mono text-[12px] text-dark-text-primary mb-4">
            {r.bestFor}
          </div>

          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-1">
            pytorch
          </div>
          <pre
            className="font-mono text-[12px] bg-dark-bg p-3 rounded border border-dark-border overflow-x-auto"
            style={{ color: r.color }}
          >
            {r.pytorch}
          </pre>
        </div>
      </div>
    </WidgetFrame>
  )
}
