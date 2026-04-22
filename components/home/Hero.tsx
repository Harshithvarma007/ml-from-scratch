'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'
import { ArrowRight } from 'lucide-react'
import Prompt from '@/components/ui/Prompt'
import { firstLesson, totalLessons } from '@/lib/roadmap'
import { lessonHref } from '@/lib/utils'
import { cn } from '@/lib/utils'

const SNIPPETS: { prompt: string; lines: { text: string; tone: 'out' | 'accent' | 'dim' }[] }[] = [
  {
    prompt: 'derive gradient-descent',
    lines: [
      { text: 'θ ← θ − η · ∇L(θ)', tone: 'accent' },
      { text: 'for step in range(epochs):', tone: 'out' },
      { text: '    w -= lr * grad(loss, w)', tone: 'out' },
      { text: '# converged in 1,284 steps ✓', tone: 'dim' },
    ],
  },
  {
    prompt: 'implement self-attention',
    lines: [
      { text: 'scores = Q @ K.T / √d_k', tone: 'accent' },
      { text: 'attn   = softmax(scores)', tone: 'out' },
      { text: 'out    = attn @ V', tone: 'out' },
      { text: '# shape: (B, T, d_model)', tone: 'dim' },
    ],
  },
  {
    prompt: 'train gpt-mini',
    lines: [
      { text: 'step   1 │ loss 4.21', tone: 'out' },
      { text: 'step 500 │ loss 2.98', tone: 'out' },
      { text: 'step 2k  │ loss 2.13 ↓', tone: 'accent' },
      { text: '# sample: "the sun rose slowly…"', tone: 'dim' },
    ],
  },
  {
    prompt: 'backprop chain-rule',
    lines: [
      { text: '∂L/∂W = (∂L/∂y) · (∂y/∂W)', tone: 'accent' },
      { text: 'dL_dW = dL_dy @ x.T', tone: 'out' },
      { text: 'dL_dx = W.T @ dL_dy', tone: 'out' },
      { text: '# gradient checked ✓', tone: 'dim' },
    ],
  },
]

export default function Hero() {
  const start = firstLesson()
  const total = totalLessons()

  return (
    <section className="relative overflow-hidden">
      {/* Background layers */}
      <div className="absolute inset-0 hero-glow pointer-events-none" />
      <div className="absolute inset-0 grid-bg pointer-events-none opacity-40 [mask-image:radial-gradient(ellipse_70%_60%_at_50%_0%,black,transparent_85%)]" />
      {/* Accent gradient */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[1px] bg-gradient-to-r from-transparent via-dark-accent/40 to-transparent" />

      <div className="relative max-w-7xl mx-auto px-6 lg:px-8 pt-24 pb-28">
        <div className="grid lg:grid-cols-2 gap-16 items-center">

          {/* Left column */}
          <div className="animate-fade-up">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-sm border border-dark-border bg-dark-surface/80 text-[11px] font-mono text-dark-text-secondary mb-8">
              <span className="w-1.5 h-1.5 rounded-full bg-term-green animate-pulse" />
              <span className="text-dark-accent">v1</span>
              <span className="text-dark-border">·</span>
              <span>{total} lessons · 14 sections · 168+ widgets</span>
            </div>

            {/* Headline */}
            <h1 className="font-mono font-light leading-[1.06] tracking-tight">
              <span className="block text-5xl md:text-6xl lg:text-7xl text-dark-text-primary">
                Derive it
              </span>
              <span className="block text-5xl md:text-6xl lg:text-7xl text-dark-accent">
                yourself.
                <span className="term-cursor" />
              </span>
            </h1>

            {/* Sub-headline */}
            <p className="mt-6 font-mono text-[13px] text-dark-text-secondary leading-relaxed max-w-lg">
              From gradient descent to GPT — every equation proven,{' '}
              every line of code written from zero,{' '}
              every research nuance unpacked to the paper.
            </p>

            {/* Stats row */}
            <div className="mt-6 flex items-center gap-4 text-[11px] font-mono">
              <StatPill value="75" label="lessons" color="text-term-purple" />
              <span className="text-dark-border">·</span>
              <StatPill value="168+" label="widgets" color="text-term-cyan" />
              <span className="text-dark-border">·</span>
              <StatPill value="14" label="sections" color="text-term-amber" />
            </div>

            {/* CTAs */}
            <div className="mt-10 flex flex-wrap items-center gap-3">
              <Link
                href={lessonHref(start.sectionSlug, start.slug)}
                className={cn(
                  'group inline-flex items-center gap-2 px-5 py-2.5 rounded-md',
                  'bg-dark-accent hover:bg-dark-accent-hover text-white',
                  'font-mono text-[13px] shadow-dark-glow hover:shadow-dark-glow-strong transition-all duration-200'
                )}
              >
                Start Deriving
                <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-0.5" />
              </Link>
              <a
                href="#curriculum"
                className={cn(
                  'inline-flex items-center gap-2 px-5 py-2.5 rounded-md',
                  'border border-dark-border bg-dark-surface hover:bg-dark-surface-hover',
                  'text-dark-text-secondary hover:text-dark-text-primary',
                  'font-mono text-[13px] transition-colors duration-200'
                )}
              >
                View Curriculum
              </a>
            </div>

            {/* Proof points */}
            <div className="mt-10 pt-8 border-t border-dark-border/40 grid grid-cols-3 gap-4 text-[11px] font-mono">
              <ProofPoint icon="∇" label="Math first" sub="every gradient derived" />
              <ProofPoint icon="⌥" label="NumPy → PyTorch" sub="three-layer code pattern" />
              <ProofPoint icon="◈" label="Live widgets" sub="poke it until it breaks" />
            </div>
          </div>

          {/* Right column — terminal */}
          <TerminalWindow />
        </div>
      </div>
    </section>
  )
}

function StatPill({ value, label, color }: { value: string; label: string; color: string }) {
  return (
    <span className="flex items-baseline gap-1">
      <span className={cn('font-semibold', color)}>{value}</span>
      <span className="text-dark-text-disabled">{label}</span>
    </span>
  )
}

function ProofPoint({ icon, label, sub }: { icon: string; label: string; sub: string }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-dark-accent text-base">{icon}</span>
      <span className="text-dark-text-secondary">{label}</span>
      <span className="text-dark-text-disabled">{sub}</span>
    </div>
  )
}

function TerminalWindow() {
  const [snippetIdx, setSnippetIdx] = useState(0)
  const [visible, setVisible] = useState(true)
  const snippet = SNIPPETS[snippetIdx]

  useEffect(() => {
    const id = setInterval(() => {
      setVisible(false)
      setTimeout(() => {
        setSnippetIdx((i) => (i + 1) % SNIPPETS.length)
        setVisible(true)
      }, 200)
    }, 4000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="relative animate-fade-up" style={{ animationDelay: '100ms' }}>
      {/* Glow halo */}
      <div className="absolute -inset-8 bg-dark-accent/8 rounded-2xl blur-3xl pointer-events-none" />

      <div className="relative term-panel rounded-xl overflow-hidden shadow-dark-lg border border-dark-border/60">
        {/* Window chrome */}
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-dark-border bg-dark-surface-elevated/60">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#ff5f57]/60" />
            <span className="w-2.5 h-2.5 rounded-full bg-[#febc2e]/60" />
            <span className="w-2.5 h-2.5 rounded-full bg-[#28c840]/60" />
          </div>
          <div className="text-[10px] font-mono text-dark-text-disabled tracking-wider">
            ml.harshithvarma.in · ~/notebook
          </div>
          <div className="text-[10px] font-mono text-dark-text-disabled tabular-nums">
            {String(snippetIdx + 1).padStart(2, '0')}/{SNIPPETS.length}
          </div>
        </div>

        {/* Accent line under chrome */}
        <div className="h-[1px] bg-gradient-to-r from-transparent via-dark-accent/30 to-transparent" />

        {/* Body */}
        <div className="p-6 bg-[#080808] min-h-[280px] font-mono text-[13px] leading-[1.75]">
          <div
            className="transition-opacity duration-200"
            style={{ opacity: visible ? 1 : 0 }}
          >
            {/* Prompt line */}
            <div className="flex items-baseline gap-2">
              <Prompt />
              <span className="text-dark-text-primary">{snippet.prompt}</span>
              <span className="term-cursor" />
            </div>
            {/* Output */}
            <div className="mt-4 space-y-1.5 pl-4 border-l border-dark-border/50">
              {snippet.lines.map((line, i) => (
                <div
                  key={i}
                  className={cn(
                    'pl-3 transition-all',
                    line.tone === 'accent' && 'text-dark-accent font-medium',
                    line.tone === 'out' && 'text-dark-text-primary',
                    line.tone === 'dim' && 'text-dark-text-muted italic'
                  )}
                >
                  {line.text}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-dark-border bg-dark-surface-elevated/40">
          <div className="flex items-center gap-1.5">
            {SNIPPETS.map((_, i) => (
              <button
                key={i}
                onClick={() => { setSnippetIdx(i); setVisible(true) }}
                className={cn(
                  'w-1.5 h-1.5 rounded-full transition-all duration-300',
                  i === snippetIdx ? 'bg-dark-accent w-3' : 'bg-dark-border-hover hover:bg-dark-text-muted'
                )}
              />
            ))}
          </div>
          <div className="text-[10px] font-mono text-dark-text-disabled">
            derive · implement · understand
          </div>
        </div>
      </div>
    </div>
  )
}
