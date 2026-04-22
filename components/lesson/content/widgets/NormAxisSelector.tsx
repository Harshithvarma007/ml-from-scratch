'use client'

import { useState } from 'react'
import WidgetFrame from './WidgetFrame'
import { cn } from '@/lib/utils'

// The 3D tensor with batch × seq × features, shown as a stack of 2D tiles.
// Toggle between LayerNorm, BatchNorm, RMSNorm. Highlight the "slice" each
// normalization averages over. Makes "which axis" a visual, not textual, fact.

type NormKind = 'layer' | 'batch' | 'rms' | 'group'

const BATCH = 4
const SEQ = 3
const FEATURES = 6

const info: Record<NormKind, { label: string; description: string; color: string; code: string }> = {
  layer: {
    label: 'LayerNorm',
    description: 'Average across features (axis=-1) for each (batch, seq) position. Independent per example.',
    color: '#a78bfa',
    code: 'nn.LayerNorm(features)',
  },
  batch: {
    label: 'BatchNorm',
    description: 'Average across batch and seq for each feature. Shared statistics across the batch.',
    color: '#67e8f9',
    code: 'nn.BatchNorm1d(features)',
  },
  rms: {
    label: 'RMSNorm',
    description: "Like LayerNorm, but skips mean subtraction. Divide by RMS only. Faster, ~equally effective.",
    color: '#fbbf24',
    code: 'nn.RMSNorm(features)  # PyTorch 2.4+',
  },
  group: {
    label: 'GroupNorm',
    description: 'Split features into G groups; normalize within each group per (batch, seq). Middle ground.',
    color: '#f472b6',
    code: 'nn.GroupNorm(num_groups=G, num_channels=features)',
  },
}

const KINDS: NormKind[] = ['layer', 'batch', 'rms', 'group']

export default function NormAxisSelector() {
  const [kind, setKind] = useState<NormKind>('layer')
  const [group, setGroup] = useState(0) // for group norm: which group is highlighted

  const current = info[kind]

  // For each (batch, seq, feature) cell, is it in the "same statistics pool" as the
  // currently-inspected anchor cell? We highlight those.
  const anchor = { b: 1, s: 1, f: 2 } // somewhat arbitrary focal cell
  const G = 2 // group count

  const isHighlighted = (b: number, s: number, f: number): boolean => {
    if (kind === 'layer' || kind === 'rms') {
      return b === anchor.b && s === anchor.s
    }
    if (kind === 'batch') {
      return f === anchor.f
    }
    if (kind === 'group') {
      const groupOf = (feat: number) => Math.floor((feat / FEATURES) * G)
      return b === anchor.b && s === anchor.s && groupOf(f) === groupOf(anchor.f)
    }
    return false
  }

  return (
    <WidgetFrame
      widgetName="NormAxisSelector"
      label="which axis does each normalization average over?"
      right={
        <>
          <span className="font-mono">tensor: (B={BATCH}, S={SEQ}, F={FEATURES})</span>
          <span className="text-dark-text-disabled">·</span>
          <span className="font-mono">{current.code}</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1">
            {KINDS.map((k) => (
              <button
                key={k}
                onClick={() => setKind(k)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono uppercase transition-all',
                  kind === k
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary'
                )}
              >
                {info[k].label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto text-[11px] font-mono text-dark-text-muted">
            anchor cell · b={anchor.b}, s={anchor.s}, f={anchor.f}
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 overflow-auto">
        <div className="max-w-[920px] mx-auto flex flex-col gap-4">
          {/* Tensor visualization — a stack of batch tiles, each showing S × F */}
          <div className="flex items-start gap-4 flex-wrap">
            {Array.from({ length: BATCH }).map((_, b) => (
              <div key={b} className="flex flex-col gap-1">
                <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
                  batch {b}
                </div>
                <div
                  className="grid gap-1 border border-dark-border p-2 rounded bg-dark-bg"
                  style={{
                    gridTemplateColumns: `18px repeat(${FEATURES}, 1fr)`,
                  }}
                >
                  <div />
                  {Array.from({ length: FEATURES }).map((_, f) => (
                    <div
                      key={f}
                      className="text-[8.5px] text-dark-text-disabled text-center font-mono"
                    >
                      f{sub(f)}
                    </div>
                  ))}
                  {Array.from({ length: SEQ }).map((_, s) => (
                    <div key={s} className="contents">
                      <div className="text-[8.5px] text-dark-text-disabled text-right font-mono pr-1 flex items-center">
                        s{sub(s)}
                      </div>
                      {Array.from({ length: FEATURES }).map((_, f) => {
                        const hot = isHighlighted(b, s, f)
                        const isAnchor = b === anchor.b && s === anchor.s && f === anchor.f
                        return (
                          <div
                            key={f}
                            className={cn(
                              'w-5 h-5 rounded transition-all',
                              hot
                                ? 'ring-1'
                                : 'bg-dark-surface-elevated/40'
                            )}
                            style={
                              hot
                                ? {
                                    backgroundColor: current.color + '66',
                                    boxShadow: isAnchor ? `inset 0 0 0 1.5px ${current.color}` : undefined,
                                  }
                                : {}
                            }
                            title={`b=${b} s=${s} f=${f}`}
                          />
                        )
                      })}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Description */}
          <div
            className="p-4 rounded border font-sans text-[13px] leading-relaxed"
            style={{
              borderColor: current.color + '66',
              backgroundColor: current.color + '0d',
              color: '#ccc',
            }}
          >
            <div
              className="font-mono uppercase tracking-wider text-[10px] mb-1"
              style={{ color: current.color }}
            >
              {current.label}
            </div>
            {current.description}
          </div>

          {/* Comparison table */}
          <div className="rounded border border-dark-border overflow-hidden">
            <table className="w-full font-mono text-[11.5px]">
              <thead className="bg-dark-surface-elevated/40 text-dark-text-disabled uppercase text-[10px]">
                <tr>
                  <th className="text-left px-3 py-2">normalization</th>
                  <th className="text-left px-3 py-2">axes averaged</th>
                  <th className="text-left px-3 py-2">stats per</th>
                  <th className="text-left px-3 py-2">needs batch?</th>
                </tr>
              </thead>
              <tbody>
                <Row highlight={kind === 'layer'} name="LayerNorm" axes="features" per="(b, s)" needsBatch="no — per sample" />
                <Row highlight={kind === 'batch'} name="BatchNorm" axes="batch + seq" per="feature" needsBatch="yes — different modes for train/eval" />
                <Row highlight={kind === 'rms'} name="RMSNorm" axes="features (no mean)" per="(b, s)" needsBatch="no" />
                <Row highlight={kind === 'group'} name="GroupNorm" axes="features in a group" per="(b, s, group)" needsBatch="no" />
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function Row({
  highlight,
  name,
  axes,
  per,
  needsBatch,
}: {
  highlight: boolean
  name: string
  axes: string
  per: string
  needsBatch: string
}) {
  return (
    <tr className={cn('border-t border-dark-border', highlight && 'bg-dark-accent/10')}>
      <td className="px-3 py-2 text-dark-text-primary">{name}</td>
      <td className="px-3 py-2 text-dark-text-secondary">{axes}</td>
      <td className="px-3 py-2 text-dark-text-secondary">{per}</td>
      <td className="px-3 py-2 text-dark-text-secondary">{needsBatch}</td>
    </tr>
  )
}

function sub(n: number): string {
  return String(n)
    .split('')
    .map((d) => '₀₁₂₃₄₅₆₇₈₉'[Number(d)] ?? d)
    .join('')
}
