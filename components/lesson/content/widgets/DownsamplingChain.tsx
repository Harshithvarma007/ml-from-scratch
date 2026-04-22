'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Classic conv-net inverted pyramid: each conv+pool block halves the spatial
// dims (H, W) while doubling the channels. Receptive field doubles per pool.
// Slider picks the current layer; stack diagram + readouts update live.

interface Layer {
  name: string
  H: number
  W: number
  C: number
  receptive: number
  kind: 'input' | 'conv' | 'pool'
}

function buildChain(inputH: number, inputC: number, blocks: number): Layer[] {
  const layers: Layer[] = [
    { name: 'input', H: inputH, W: inputH, C: inputC, receptive: 1, kind: 'input' },
  ]
  for (let k = 1; k <= blocks; k++) {
    const prev = layers[layers.length - 1]
    layers.push({
      name: `conv${k}`,
      H: prev.H,
      W: prev.W,
      C: prev.C * 2,
      receptive: prev.receptive + 2,
      kind: 'conv',
    })
    const conv = layers[layers.length - 1]
    layers.push({
      name: `pool${k}`,
      H: Math.max(1, Math.floor(conv.H / 2)),
      W: Math.max(1, Math.floor(conv.W / 2)),
      C: conv.C,
      receptive: conv.receptive * 2,
      kind: 'pool',
    })
  }
  return layers
}

export default function DownsamplingChain() {
  const layers = useMemo(() => buildChain(64, 3, 4), [])
  const [cursor, setCursor] = useState(layers.length - 1)
  const maxHW = Math.max(...layers.map((l) => l.H))

  const l = layers[cursor]

  return (
    <WidgetFrame
      widgetName="DownsamplingChain"
      label="downsampling chain — spatial dims shrink, channels grow"
      right={
        <span className="font-mono">
          conv doubles C · pool halves H,W · receptive field doubles per pool
        </span>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="layer"
            value={cursor}
            min={0}
            max={layers.length - 1}
            step={1}
            onChange={(v) => setCursor(Math.round(v))}
            format={(v) => layers[Math.round(v)]?.name ?? '—'}
            accent="accent-term-cyan"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="shape"
              value={`${l.C} × ${l.H} × ${l.W}`}
              accent="text-term-amber"
            />
            <Readout
              label="receptive field"
              value={`${l.receptive} × ${l.receptive}`}
              accent="text-term-purple"
            />
            <Readout label="#params" value={cursor === 0 ? '—' : estimateParams(layers, cursor)} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 flex items-center overflow-auto">
        <div className="flex items-end gap-4 min-w-max mx-auto">
          {layers.map((layer, i) => {
            const active = i <= cursor
            const spatial = 18 + (layer.H / maxHW) * 80
            const channelsBars = Math.min(10, Math.max(2, Math.round(Math.log2(layer.C))))
            return (
              <div
                key={i}
                className={cn(
                  'flex flex-col items-center gap-1.5 transition-all',
                  !active && 'opacity-20',
                )}
              >
                <div
                  className={cn(
                    'text-[10px] font-mono uppercase tracking-wider',
                    layer.kind === 'pool'
                      ? 'text-term-amber'
                      : layer.kind === 'conv'
                        ? 'text-term-purple'
                        : 'text-term-cyan',
                  )}
                >
                  {layer.name}
                </div>
                <div
                  className="flex items-end gap-[1px]"
                  style={{ width: `${spatial}px`, height: `${spatial}px` }}
                >
                  {Array.from({ length: channelsBars }).map((_, c) => (
                    <div
                      key={c}
                      className={cn(
                        'flex-1 rounded-sm',
                        layer.kind === 'pool'
                          ? 'bg-term-amber/50 ring-1 ring-term-amber/70'
                          : layer.kind === 'conv'
                            ? 'bg-term-purple/50 ring-1 ring-term-purple/70'
                            : 'bg-term-cyan/50 ring-1 ring-term-cyan/70',
                      )}
                    />
                  ))}
                </div>
                <div className="text-[10px] font-mono text-dark-text-muted tabular-nums">
                  {layer.C}×{layer.H}×{layer.W}
                </div>
                <div className="text-[9px] font-mono text-dark-text-disabled tabular-nums">
                  RF {layer.receptive}
                </div>
              </div>
            )
          })}
        </div>
      </div>
      <div className="absolute bottom-2 left-4 text-[10.5px] font-mono text-dark-text-disabled pointer-events-none flex items-center gap-4">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 bg-term-cyan/60 rounded-sm" /> input
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 bg-term-purple/60 rounded-sm" /> conv (3×3, +C)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 bg-term-amber/60 rounded-sm" /> max-pool (2×2, /HW)
        </span>
      </div>
    </WidgetFrame>
  )
}

// Rough #params up to cursor assuming 3x3 conv (9*C_in*C_out + C_out per conv).
function estimateParams(layers: Layer[], idx: number): string {
  let total = 0
  for (let i = 1; i <= idx; i++) {
    const prev = layers[i - 1]
    const cur = layers[i]
    if (cur.kind === 'conv') total += 9 * prev.C * cur.C + cur.C
  }
  if (total < 1000) return String(total)
  if (total < 1e6) return (total / 1000).toFixed(1) + 'K'
  return (total / 1e6).toFixed(2) + 'M'
}
