'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Three hand-authored preference pairs. The user clicks "A" or "B" to label
// each. We then compute the Bradley-Terry RM loss on the labeled data:
// L = -log σ(r_chosen - r_rejected), averaged. Dummy reward scores per
// response are fixed (so clicking rewards the model with a consistent
// decrease in loss when the "better" one is picked).

type Pair = {
  id: number
  prompt: string
  a: string
  b: string
  rA: number
  rB: number
  // Which is the "ground-truth" preferred one (hand-picked by the authors).
  correct: 'A' | 'B'
}

const PAIRS: Pair[] = [
  {
    id: 1,
    prompt: 'How do I sort a list in Python?',
    a: 'Use list.sort() for in-place, or sorted(list) for a new copy. Both accept a key= argument.',
    b: 'idk try asking google or something',
    rA: 1.8,
    rB: -1.2,
    correct: 'A',
  },
  {
    id: 2,
    prompt: 'Write a haiku about debugging.',
    a: 'Errors stack like snow\nI refresh and pray — the build\nFinally compiles.',
    b: 'debug is hard. i cant always fix it. sometimes code works sometimes it dont.',
    rA: 1.4,
    rB: -0.6,
    correct: 'A',
  },
  {
    id: 3,
    prompt: 'Explain backpropagation in one sentence.',
    a: 'Backpropagation computes gradients of the loss with respect to each parameter by applying the chain rule backward through the computation graph.',
    b: 'Backprop is when your neural network gets angry and tells all its friends.',
    rA: 1.1,
    rB: -0.4,
    correct: 'A',
  },
]

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

export default function PreferencePair() {
  // chosen[i] = 'A' | 'B' | null (unlabeled)
  const [chosen, setChosen] = useState<Record<number, 'A' | 'B' | null>>({ 1: null, 2: null, 3: null })

  const { loss, correctCount, labeledCount } = useMemo(() => {
    let totalLoss = 0
    let correctN = 0
    let labeledN = 0
    PAIRS.forEach((p) => {
      const c = chosen[p.id]
      if (!c) return
      labeledN++
      const rC = c === 'A' ? p.rA : p.rB
      const rR = c === 'A' ? p.rB : p.rA
      totalLoss += -Math.log(Math.max(1e-9, sigmoid(rC - rR)))
      if (c === p.correct) correctN++
    })
    return {
      loss: labeledN > 0 ? totalLoss / labeledN : NaN,
      correctCount: correctN,
      labeledCount: labeledN,
    }
  }, [chosen])

  const reset = () => setChosen({ 1: null, 2: null, 3: null })

  return (
    <WidgetFrame
      widgetName="PreferencePair"
      label="preference pairs — label the winner, watch the RM loss"
      right={<span className="font-mono">L = −log σ(r_chosen − r_rejected)</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <button
            onClick={reset}
            className="px-2.5 py-1 rounded text-[11px] font-mono border border-dark-border text-dark-text-secondary hover:text-dark-text-primary"
          >
            reset labels
          </button>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="labeled" value={`${labeledCount} / ${PAIRS.length}`} accent="text-term-amber" />
            <Readout label="agreement" value={labeledCount > 0 ? `${correctCount} / ${labeledCount}` : '—'} accent="text-term-cyan" />
            <Readout label="mean loss" value={isNaN(loss) ? '—' : loss.toFixed(3)} accent={loss < 0.1 ? 'text-term-green' : 'text-term-amber'} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-auto flex flex-col gap-3">
        {PAIRS.map((p) => {
          const c = chosen[p.id]
          return (
            <div key={p.id} className="rounded border border-dark-border bg-dark-surface-elevated/30 p-3">
              <div className="font-mono text-[10px] uppercase tracking-wider text-dark-text-disabled mb-1">
                prompt #{p.id}
              </div>
              <div className="font-mono text-[12px] text-dark-text-primary mb-2">{p.prompt}</div>
              <div className="grid grid-cols-2 gap-2">
                <ResponseCard
                  letter="A"
                  text={p.a}
                  score={p.rA}
                  picked={c === 'A'}
                  rejected={c === 'B'}
                  onClick={() => setChosen({ ...chosen, [p.id]: 'A' })}
                />
                <ResponseCard
                  letter="B"
                  text={p.b}
                  score={p.rB}
                  picked={c === 'B'}
                  rejected={c === 'A'}
                  onClick={() => setChosen({ ...chosen, [p.id]: 'B' })}
                />
              </div>

              {c && (
                <div className="flex items-center justify-between mt-2 pt-2 border-t border-dark-border font-mono text-[10.5px]">
                  <span className="text-dark-text-muted">
                    r_chosen = <span className="text-term-green">{(c === 'A' ? p.rA : p.rB).toFixed(2)}</span>
                    <span className="mx-1 text-dark-text-disabled">·</span>
                    r_rejected = <span className="text-term-rose">{(c === 'A' ? p.rB : p.rA).toFixed(2)}</span>
                  </span>
                  <span className={cn('text-right', c === p.correct ? 'text-term-green' : 'text-term-rose')}>
                    pair loss = {(-Math.log(Math.max(1e-9, sigmoid((c === 'A' ? p.rA : p.rB) - (c === 'A' ? p.rB : p.rA))))).toFixed(3)}
                  </span>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </WidgetFrame>
  )
}

function ResponseCard({
  letter,
  text,
  score,
  picked,
  rejected,
  onClick,
}: {
  letter: string
  text: string
  score: number
  picked: boolean
  rejected: boolean
  onClick: () => void
}) {
  // Bar is centered, ranges [-3, +3] visually.
  const pct = Math.max(-3, Math.min(3, score)) / 3
  return (
    <button
      onClick={onClick}
      className={cn(
        'flex flex-col items-stretch gap-1.5 p-2 rounded border text-left font-mono text-[11px] transition-all',
        picked && 'border-term-green bg-term-green/10 text-term-green',
        rejected && 'border-term-rose/50 bg-term-rose/5 text-term-rose/80',
        !picked && !rejected && 'border-dark-border bg-dark-bg text-dark-text-primary hover:border-dark-border-hover',
      )}
    >
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-dark-text-disabled">response {letter}</span>
        <span className="tabular-nums">r = {score.toFixed(2)}</span>
      </div>
      <div className="text-[11px] leading-snug">{text}</div>
      <div className="relative h-2 bg-dark-surface-elevated/40 rounded-sm overflow-hidden mt-1">
        <div className="absolute top-0 bottom-0 border-l border-dark-border" style={{ left: '50%' }} />
        <div
          className={cn('absolute top-0 bottom-0', score >= 0 ? 'bg-term-green/70' : 'bg-term-rose/70')}
          style={{
            left: score >= 0 ? '50%' : `${50 + pct * 50}%`,
            width: `${Math.abs(pct) * 50}%`,
          }}
        />
      </div>
    </button>
  )
}
