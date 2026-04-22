'use client'

import { useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Gallery of 12 stylized "CIFAR-10" images drawn procedurally. Each has
// hand-curated top-5 predictions that look like a real 94%-accurate net —
// most are confident and correct, a couple are classic confusions.

const CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

type Sample = {
  label: number
  preds: number[] // 10-length
  // Simple SVG icon instructions: scenes that hint at the class
  icon: 'plane' | 'car' | 'bird' | 'cat' | 'deer' | 'dog' | 'frog' | 'horse' | 'ship' | 'truck'
  hint: string
}

function distribution(trueIdx: number, topNoise: Array<[number, number]>): number[] {
  const logits = new Array(10).fill(-3)
  logits[trueIdx] = 5
  topNoise.forEach(([i, v]) => (logits[i] = v))
  const m = Math.max(...logits)
  const e = logits.map((v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

// Hand-tuned samples
const SAMPLES: Sample[] = [
  { icon: 'plane', label: 0, preds: distribution(0, [[8, 1]]), hint: 'sky + wings · confident' },
  { icon: 'car', label: 1, preds: distribution(1, [[9, 2]]), hint: 'car vs truck — close call' },
  { icon: 'bird', label: 2, preds: distribution(2, [[0, 0.5]]), hint: 'bird · clear' },
  { icon: 'cat', label: 3, preds: distribution(3, [[5, 3]]), hint: 'cat/dog ambiguity' },
  { icon: 'deer', label: 4, preds: distribution(4, [[7, 1]]), hint: 'deer vs horse' },
  { icon: 'dog', label: 5, preds: distribution(5, [[3, 2]]), hint: 'dog · some cat' },
  { icon: 'frog', label: 6, preds: distribution(6, []), hint: 'frog · no confusion' },
  { icon: 'horse', label: 7, preds: distribution(7, [[4, 0.8]]), hint: 'horse · deer in tail' },
  { icon: 'ship', label: 8, preds: distribution(8, [[0, 1.2]]), hint: 'ship vs plane at sea' },
  { icon: 'truck', label: 9, preds: distribution(9, [[1, 2.5]]), hint: 'truck / car visually similar' },
  { icon: 'cat', label: 3, preds: distribution(5, [[3, 4]]), hint: 'mislabeled · actually cat, predicted dog' },
  { icon: 'bird', label: 2, preds: distribution(0, [[2, 3]]), hint: 'bird against sky · misreads as plane' },
]

export default function CIFARPredictions() {
  const [sel, setSel] = useState(0)
  const sample = SAMPLES[sel]

  const correct = sample.preds.indexOf(Math.max(...sample.preds)) === sample.label
  const top5 = [...sample.preds.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([idx, p]) => ({ idx, p }))

  return (
    <WidgetFrame
      widgetName="CIFARPredictions"
      label="CIFAR-10 classifier — curated predictions"
      right={<span className="font-mono">pretend ResNet-18 · 94% test acc</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="text-[11px] font-mono text-dark-text-muted">
            click an image to see top-5 predictions
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout
              label="top-1"
              value={CLASSES[top5[0].idx]}
              accent={correct ? 'text-term-green' : 'text-term-rose'}
            />
            <Readout
              label="confidence"
              value={`${(top5[0].p * 100).toFixed(0)}%`}
            />
            <Readout
              label="correct?"
              value={correct ? 'yes' : 'no'}
              accent={correct ? 'text-term-green' : 'text-term-rose'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 grid grid-cols-1 md:grid-cols-[1fr_320px] gap-4 overflow-auto">
        {/* Gallery */}
        <div className="grid grid-cols-6 gap-2 auto-rows-min">
          {SAMPLES.map((s, i) => (
            <button
              key={i}
              onClick={() => setSel(i)}
              className={cn(
                'aspect-square rounded border transition-all relative',
                sel === i
                  ? 'border-dark-accent ring-2 ring-dark-accent/60'
                  : 'border-dark-border hover:border-dark-border-hover',
              )}
              title={`actual: ${CLASSES[s.label]}`}
            >
              <SampleIcon icon={s.icon} />
              <span className="absolute bottom-0.5 left-0.5 text-[8.5px] font-mono text-dark-text-disabled bg-dark-bg/70 px-1 rounded">
                {CLASSES[s.label]}
              </span>
            </button>
          ))}
        </div>

        {/* Top-5 bars */}
        <div className="rounded border border-dark-border bg-dark-bg p-4 self-start">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mb-1">
            top 5 predictions
          </div>
          <div className="text-[10.5px] font-mono text-dark-text-muted mb-3">
            true class: <span className="text-term-green">{CLASSES[sample.label]}</span> · {sample.hint}
          </div>
          <div className="space-y-1.5">
            {top5.map(({ idx, p }, rank) => (
              <div key={rank} className="flex items-center gap-2 font-mono text-[11px]">
                <span className="w-12 text-dark-text-secondary">{CLASSES[idx]}</span>
                <div className="flex-1 h-4 bg-dark-surface-elevated/40 rounded overflow-hidden">
                  <div
                    className={cn(
                      'h-full transition-all',
                      idx === sample.label ? 'bg-term-green/70' : 'bg-term-amber/60',
                    )}
                    style={{ width: `${(p * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="w-10 text-right tabular-nums text-dark-text-muted">
                  {(p * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

// Minimal procedural icons for each class
function SampleIcon({ icon }: { icon: Sample['icon'] }) {
  const common = { stroke: '#e5e7eb', strokeWidth: 1.2, fill: 'none' }
  return (
    <svg viewBox="0 0 32 32" className="w-full h-full p-1.5">
      <rect x="0" y="0" width="32" height="32" fill="#0f1225" />
      {icon === 'plane' && (
        <>
          <path d="M4 17 L28 15 L20 17 L19 22 M4 17 L19 18" {...common} />
          <circle cx="16" cy="17" r="0.5" fill="#e5e7eb" />
        </>
      )}
      {icon === 'car' && (
        <>
          <path d="M4 22 L8 18 L24 18 L28 22 Z" {...common} />
          <circle cx="10" cy="23" r="2" {...common} />
          <circle cx="22" cy="23" r="2" {...common} />
        </>
      )}
      {icon === 'bird' && (
        <>
          <path d="M8 20 Q16 10 24 20 Q22 16 16 18 Q10 16 8 20" {...common} />
        </>
      )}
      {icon === 'cat' && (
        <>
          <circle cx="16" cy="18" r="6" {...common} />
          <path d="M11 15 L10 12 L13 14 M21 15 L22 12 L19 14" {...common} />
          <circle cx="14" cy="18" r="0.6" fill="#e5e7eb" />
          <circle cx="18" cy="18" r="0.6" fill="#e5e7eb" />
        </>
      )}
      {icon === 'deer' && (
        <>
          <path d="M10 12 L8 8 M12 12 L11 7 L14 10 M22 12 L24 8 M20 12 L21 7 L18 10" {...common} />
          <circle cx="16" cy="18" r="4" {...common} />
        </>
      )}
      {icon === 'dog' && (
        <>
          <circle cx="16" cy="18" r="6" {...common} />
          <path d="M10 13 L9 17 M22 13 L23 17" {...common} />
          <circle cx="14" cy="18" r="0.6" fill="#e5e7eb" />
          <circle cx="18" cy="18" r="0.6" fill="#e5e7eb" />
        </>
      )}
      {icon === 'frog' && (
        <>
          <ellipse cx="16" cy="20" rx="8" ry="5" {...common} />
          <circle cx="12" cy="16" r="2" {...common} />
          <circle cx="20" cy="16" r="2" {...common} />
        </>
      )}
      {icon === 'horse' && (
        <>
          <path d="M6 22 L8 12 L14 10 L16 18 L24 22 L22 14 L18 10 L16 10" {...common} />
        </>
      )}
      {icon === 'ship' && (
        <>
          <path d="M4 22 L28 22 L26 18 L6 18 Z" {...common} />
          <path d="M16 18 L16 6 L22 14 L16 14" {...common} />
        </>
      )}
      {icon === 'truck' && (
        <>
          <path d="M4 20 L4 14 L18 14 L18 10 L26 10 L28 14 L28 20 Z" {...common} />
          <circle cx="10" cy="22" r="2" {...common} />
          <circle cx="24" cy="22" r="2" {...common} />
        </>
      )}
    </svg>
  )
}
