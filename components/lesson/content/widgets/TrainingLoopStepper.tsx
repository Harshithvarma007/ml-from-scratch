'use client'

import { useState } from 'react'
import WidgetFrame, { Button } from './WidgetFrame'
import { Play, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'

// Step through the 5-line training loop, one line at a time. Watch exactly
// four pieces of state: the weight tensor, its .grad buffer, the forward
// output, and the loss. That is the entire mental model of a PyTorch step.

interface State {
  w: number
  b: number
  grad_w: number | null
  grad_b: number | null
  yhat: number | null
  loss: number | null
  stepPhase: 'idle' | 'zero_grad' | 'forward' | 'loss' | 'backward' | 'step' | 'done'
}

// A single training step on:  L = ((w * 2 + b) - 1.0)^2  with (w, b) starting at (0.2, 0.1).
const LR = 0.3

function zeroGrad(s: State): State {
  return { ...s, grad_w: 0, grad_b: 0, stepPhase: 'zero_grad' }
}
function forward(s: State): State {
  const yhat = s.w * 2 + s.b
  return { ...s, yhat, stepPhase: 'forward' }
}
function computeLoss(s: State): State {
  if (s.yhat === null) return s
  const loss = (s.yhat - 1.0) ** 2
  return { ...s, loss, stepPhase: 'loss' }
}
function backward(s: State): State {
  if (s.yhat === null) return s
  // dL/dw = 2(yhat - 1) * 2 ;  dL/db = 2(yhat - 1)
  const diff = s.yhat - 1.0
  return {
    ...s,
    grad_w: (s.grad_w ?? 0) + 2 * diff * 2,
    grad_b: (s.grad_b ?? 0) + 2 * diff,
    stepPhase: 'backward',
  }
}
function step(s: State): State {
  if (s.grad_w === null || s.grad_b === null) return s
  return {
    ...s,
    w: s.w - LR * s.grad_w,
    b: s.b - LR * s.grad_b,
    stepPhase: 'step',
  }
}

const PHASES: Array<(s: State) => State> = [zeroGrad, forward, computeLoss, backward, step]
const PHASE_LABELS = [
  'optimizer.zero_grad()',
  'yhat = model(x)',
  'loss = criterion(yhat, y)',
  'loss.backward()',
  'optimizer.step()',
]
const PHASE_EXPLAIN = [
  'Clear the gradient buffer. Leftover grads from the previous step would accumulate otherwise.',
  'Run the forward pass. PyTorch silently builds the computation graph along the way.',
  'Compute the scalar loss. Autograd now knows exactly what it needs to differentiate.',
  'Walk the graph in reverse. For each parameter, fill .grad with dL/dparam.',
  'Apply the update rule. θ ← θ − α·∇L. One step of gradient descent, done.',
]

const INITIAL: State = {
  w: 0.2,
  b: 0.1,
  grad_w: null,
  grad_b: null,
  yhat: null,
  loss: null,
  stepPhase: 'idle',
}

export default function TrainingLoopStepper() {
  const [state, setState] = useState<State>(INITIAL)
  const [stepIdx, setStepIdx] = useState(0)

  const nextLine = () => {
    if (stepIdx >= PHASES.length) {
      // Start a new outer iteration
      setStepIdx(1)
      setState((s) => forward(zeroGrad(s)))
      return
    }
    setState(PHASES[stepIdx])
    setStepIdx((i) => i + 1)
  }

  const reset = () => {
    setState(INITIAL)
    setStepIdx(0)
  }

  const curLine = stepIdx === 0 ? -1 : stepIdx - 1

  return (
    <WidgetFrame
      widgetName="TrainingLoopStepper"
      label="the 5-line training loop, one line at a time"
      right={
        <>
          <span className="font-mono">target y = 1.0 · lr α = {LR}</span>
        </>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Button onClick={nextLine} variant="primary">
            <Play className="w-3 h-3 inline -mt-px mr-1" />{' '}
            {stepIdx === 0 ? 'run line 1' : stepIdx >= PHASES.length ? 'next iteration' : `run line ${stepIdx + 1}`}
          </Button>
          <Button onClick={reset}>
            <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
          </Button>
          <span className="text-[11px] font-mono text-dark-text-muted">
            phase: {state.stepPhase}
          </span>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 overflow-auto">
        <div className="max-w-[920px] mx-auto grid grid-cols-[1fr_1fr] gap-4">
          {/* Code column */}
          <div className="border border-dark-border rounded bg-dark-bg p-4 font-mono text-[12px]">
            <div className="text-[10px] uppercase tracking-wider text-dark-text-disabled mb-2">
              pytorch training loop
            </div>
            {PHASE_LABELS.map((line, i) => (
              <div
                key={i}
                className={cn(
                  'px-2 py-1 rounded mb-0.5 transition-all',
                  i === curLine
                    ? 'bg-dark-accent/15 text-dark-text-primary shadow-[inset_2px_0_0_#a78bfa]'
                    : i < curLine
                      ? 'text-dark-text-muted'
                      : 'text-dark-text-disabled'
                )}
              >
                {line}
              </div>
            ))}
            {curLine >= 0 && (
              <div className="mt-3 pt-3 border-t border-dark-border text-[11px] font-sans text-dark-text-secondary leading-relaxed">
                <span className="text-dark-accent font-mono text-[10px] uppercase tracking-wider">
                  what it does:{' '}
                </span>
                {PHASE_EXPLAIN[curLine]}
              </div>
            )}
          </div>

          {/* State column */}
          <div className="border border-dark-border rounded bg-dark-bg p-4 font-mono text-[12px] space-y-3">
            <div className="text-[10px] uppercase tracking-wider text-dark-text-disabled">
              tensor state
            </div>
            <StateRow name="w.data" value={state.w.toFixed(4)} color="#67e8f9" />
            <StateRow
              name="w.grad"
              value={state.grad_w === null ? 'None' : state.grad_w.toFixed(4)}
              color="#f472b6"
              changed={state.stepPhase === 'backward' || state.stepPhase === 'zero_grad'}
            />
            <StateRow name="b.data" value={state.b.toFixed(4)} color="#67e8f9" />
            <StateRow
              name="b.grad"
              value={state.grad_b === null ? 'None' : state.grad_b.toFixed(4)}
              color="#f472b6"
              changed={state.stepPhase === 'backward' || state.stepPhase === 'zero_grad'}
            />
            <div className="border-t border-dark-border pt-2" />
            <StateRow
              name="yhat"
              value={state.yhat === null ? '—' : state.yhat.toFixed(4)}
              color="#fbbf24"
              changed={state.stepPhase === 'forward'}
            />
            <StateRow
              name="loss"
              value={state.loss === null ? '—' : state.loss.toFixed(4)}
              color="#fbbf24"
              changed={state.stepPhase === 'loss'}
            />
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function StateRow({
  name,
  value,
  color,
  changed,
}: {
  name: string
  value: string
  color: string
  changed?: boolean
}) {
  return (
    <div
      className={cn(
        'flex items-center justify-between px-2 py-1 rounded transition-colors',
        changed && 'bg-dark-accent/10'
      )}
    >
      <span className="text-dark-text-muted">{name}</span>
      <span className="tabular-nums" style={{ color }}>
        {value}
      </span>
    </div>
  )
}
