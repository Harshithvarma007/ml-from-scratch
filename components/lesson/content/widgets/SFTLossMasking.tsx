'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Show an SFT example broken into tokens and visualize which positions
// contribute to the loss. Prompt tokens are dim (masked); response tokens are
// amber (loss applied). A click toggles any token. A mode switch shows the
// broken alternative — "train on full sequence" — where the model is also
// penalized for predicting the user's own question, which teaches it to
// parrot prompts.

const EXAMPLE_PROMPT_TOKENS = [
  '<|im_start|>', 'user', '\\n',
  'Write', ' a', ' haiku', ' about', ' rain', '.',
  '<|im_end|>', '\\n',
  '<|im_start|>', 'assistant', '\\n',
]
const EXAMPLE_RESPONSE_TOKENS = [
  'Silver', ' threads', ' fall', ',',
  '\\n', 'gray', ' clouds', ' whispering', ' lullabies', '—',
  '\\n', 'earth', ' drinks', ' gratefully', '.',
  '<|im_end|>',
]

type Mode = 'standard' | 'full'

export default function SFTLossMasking() {
  const initial = useMemo(() => {
    const m: boolean[] = []
    EXAMPLE_PROMPT_TOKENS.forEach(() => m.push(false))
    EXAMPLE_RESPONSE_TOKENS.forEach(() => m.push(true))
    return m
  }, [])
  const tokens = useMemo(
    () => [...EXAMPLE_PROMPT_TOKENS, ...EXAMPLE_RESPONSE_TOKENS],
    [],
  )

  const [mask, setMask] = useState<boolean[]>(initial)
  const [mode, setMode] = useState<Mode>('standard')

  const effective = mode === 'standard' ? mask : tokens.map(() => true)
  const lossCount = effective.filter(Boolean).length
  const total = tokens.length
  const coverage = (lossCount / total) * 100

  const toggle = (i: number) => {
    if (mode === 'full') return
    const next = mask.slice()
    next[i] = !next[i]
    setMask(next)
  }

  const reset = () => setMask(initial)

  return (
    <WidgetFrame
      widgetName="SFTLossMasking"
      label="loss masking — what the optimizer actually sees"
      right={<span className="font-mono">click a token to toggle its mask</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setMode('standard')}
              className={cn(
                'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
                mode === 'standard' ? 'bg-dark-accent text-white' : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
              )}
            >
              mask prompt (standard)
            </button>
            <button
              onClick={() => setMode('full')}
              className={cn(
                'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
                mode === 'full' ? 'bg-term-rose/70 text-white' : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
              )}
            >
              train on full sequence (wrong)
            </button>
            <button
              onClick={reset}
              className="px-2.5 py-1 rounded text-[11px] font-mono border border-dark-border text-dark-text-secondary hover:text-dark-text-primary"
            >
              reset
            </button>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="loss tokens" value={`${lossCount} / ${total}`} accent="text-term-amber" />
            <Readout label="coverage" value={`${coverage.toFixed(0)}%`} accent={coverage > 60 ? 'text-term-rose' : 'text-term-green'} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden grid grid-rows-[auto_1fr_auto] gap-3">
        <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
          conversation, tokenized — amber = counted in loss · dim = masked (skipped)
        </div>

        <div className="flex flex-wrap gap-1 content-start font-mono text-[12px] leading-tight overflow-auto">
          {tokens.map((t, i) => {
            const active = effective[i]
            const isSpecial = t.startsWith('<|') || t === '\\n'
            return (
              <button
                key={i}
                onClick={() => toggle(i)}
                className={cn(
                  'px-1.5 py-0.5 rounded transition-all border',
                  active
                    ? 'bg-term-amber/15 border-term-amber/40 text-term-amber'
                    : 'bg-dark-surface-elevated/40 border-dark-border text-dark-text-disabled',
                  mode === 'full' ? 'cursor-not-allowed' : 'cursor-pointer hover:border-dark-border-hover',
                  isSpecial && active && 'text-term-pink border-term-pink/40 bg-term-pink/10',
                  isSpecial && !active && 'text-dark-text-muted',
                )}
                title={`token ${i}: ${JSON.stringify(t)} — ${active ? 'in loss' : 'masked'}`}
              >
                {t === '\\n' ? '↵' : t.replace(/\s/g, '·')}
              </button>
            )
          })}
        </div>

        <div className={cn(
          'font-mono text-[11px] leading-snug px-3 py-2 rounded border',
          mode === 'standard'
            ? 'text-dark-text-muted border-term-green/30 bg-term-green/5'
            : 'text-term-rose border-term-rose/40 bg-term-rose/5',
        )}>
          {mode === 'standard' ? (
            <>
              <span className="text-term-green">standard SFT:</span> cross-entropy only over assistant tokens. The model
              learns to produce the response, not to parrot the user&apos;s prompt.
            </>
          ) : (
            <>
              <span className="text-term-rose">broken:</span> training on every position makes the model fit the user&apos;s
              question too — at inference it will happily generate user turns back. Coverage bloats, signal dilutes.
            </>
          )}
        </div>
      </div>
    </WidgetFrame>
  )
}
