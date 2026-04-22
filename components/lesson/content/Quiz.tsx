'use client'

import { useState, type ReactNode } from 'react'
import { Check, HelpCircle, X } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface QuizOption {
  /** The option text shown on the button. */
  text: ReactNode
  /** Whether this option is the correct answer (or one of several). */
  correct?: boolean
  /** Reveal-on-click explainer. Renders underneath the option after any pick. */
  explain?: ReactNode
}

interface QuizProps {
  /** The question prompt. Short, one sentence preferred. */
  question: ReactNode
  /** Between 2 and 5 options. Order is rendered verbatim — shuffle at the call site if you want it. */
  options: QuizOption[]
  /** Override the default "quiz" header label. */
  label?: string
}

// Lightweight end-of-section comprehension check. Pedagogy: rather than
// hiding the answer behind a "reveal" button, we let the reader commit to a
// guess, then show the explanation for *every* option — including the one
// they didn't pick — so a wrong guess still teaches something.
//
// Local state only; no persistence across reloads on purpose. A reader who
// comes back tomorrow should re-engage, not skim their old answers.
export default function Quiz({ question, options, label = 'quiz' }: QuizProps) {
  const [picked, setPicked] = useState<number | null>(null)
  const revealed = picked !== null

  return (
    <div className="term-panel rounded-lg overflow-hidden my-6" role="group" aria-label="Quick check">
      <header className="term-panel-header">
        <div className="flex items-center gap-2">
          <HelpCircle className="w-3 h-3 text-dark-accent" strokeWidth={2} />
          <span>{label}</span>
        </div>
        {revealed && (
          <button
            type="button"
            onClick={() => setPicked(null)}
            className="text-[10px] normal-case tracking-normal text-dark-text-disabled hover:text-dark-text-primary font-mono transition-colors"
          >
            reset
          </button>
        )}
      </header>

      <div className="p-4 space-y-3">
        <p className="font-sans text-[13.5px] leading-relaxed text-dark-text-primary">
          {question}
        </p>

        <ul className="space-y-2" role="radiogroup" aria-label="Answer options">
          {options.map((opt, i) => {
            const isPicked = picked === i
            const isCorrect = !!opt.correct
            const state: 'idle' | 'correct' | 'wrong' | 'unpicked-correct' | 'unpicked' =
              !revealed
                ? 'idle'
                : isPicked && isCorrect
                  ? 'correct'
                  : isPicked && !isCorrect
                    ? 'wrong'
                    : !isPicked && isCorrect
                      ? 'unpicked-correct'
                      : 'unpicked'

            return (
              <li key={i}>
                <button
                  type="button"
                  role="radio"
                  aria-checked={isPicked}
                  disabled={revealed}
                  onClick={() => setPicked(i)}
                  className={cn(
                    'w-full text-left font-sans text-[13px] leading-relaxed',
                    'px-3 py-2 rounded-md border transition-all',
                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-dark-accent focus-visible:ring-offset-2 focus-visible:ring-offset-dark-bg',
                    !revealed &&
                      'border-dark-border bg-dark-surface hover:bg-dark-surface-elevated hover:border-dark-border-hover text-dark-text-primary cursor-pointer',
                    state === 'correct' &&
                      'border-term-green/60 bg-term-green/[0.08] text-dark-text-primary',
                    state === 'wrong' &&
                      'border-term-rose/60 bg-term-rose/[0.08] text-dark-text-primary',
                    state === 'unpicked-correct' &&
                      'border-term-green/40 bg-term-green/[0.04] text-dark-text-secondary',
                    state === 'unpicked' &&
                      'border-dark-border bg-dark-surface/60 text-dark-text-disabled',
                    revealed && 'cursor-default',
                  )}
                >
                  <div className="flex items-start gap-2">
                    <span
                      className={cn(
                        'mt-0.5 inline-flex w-4 h-4 rounded-full border items-center justify-center shrink-0 text-[10px] font-mono',
                        state === 'correct' && 'border-term-green text-term-green',
                        state === 'wrong' && 'border-term-rose text-term-rose',
                        state === 'unpicked-correct' &&
                          'border-term-green/60 text-term-green/80',
                        (state === 'idle' || state === 'unpicked') &&
                          'border-dark-border text-dark-text-disabled',
                      )}
                      aria-hidden="true"
                    >
                      {state === 'correct' || state === 'unpicked-correct' ? (
                        <Check className="w-2.5 h-2.5" strokeWidth={3} />
                      ) : state === 'wrong' ? (
                        <X className="w-2.5 h-2.5" strokeWidth={3} />
                      ) : (
                        String.fromCharCode(65 + i)
                      )}
                    </span>
                    <span className="flex-1">{opt.text}</span>
                  </div>
                  {revealed && opt.explain && (
                    <p className="mt-2 pl-6 text-[12.5px] text-dark-text-secondary leading-relaxed">
                      {opt.explain}
                    </p>
                  )}
                </button>
              </li>
            )
          })}
        </ul>

        {revealed && (
          <p
            className={cn(
              'font-mono text-[11px] uppercase tracking-wider pt-1',
              options[picked!].correct ? 'text-term-green' : 'text-term-rose',
            )}
            role="status"
            aria-live="polite"
          >
            {options[picked!].correct
              ? '✓ correct — read the others to see why they\u2019re not'
              : '✗ not quite — the green row has the right answer'}
          </p>
        )}
      </div>
    </div>
  )
}
