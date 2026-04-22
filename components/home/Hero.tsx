'use client'

import Link from 'next/link'
import { ArrowRight } from 'lucide-react'
import { firstLesson, totalLessons } from '@/lib/roadmap'
import { lessonHref } from '@/lib/utils'
import { cn } from '@/lib/utils'

export default function Hero() {
  const start = firstLesson()
  const total = totalLessons()

  return (
    <section className="relative overflow-hidden">
      <div className="absolute inset-0 hero-glow pointer-events-none" />
      <div className="absolute inset-0 grid-bg pointer-events-none opacity-40 [mask-image:radial-gradient(ellipse_70%_60%_at_50%_0%,black,transparent_85%)]" />
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[1px] bg-gradient-to-r from-transparent via-dark-accent/40 to-transparent" />

      <div className="relative max-w-5xl mx-auto px-6 lg:px-8 pt-28 pb-28">
        <div className="animate-fade-up max-w-3xl">
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

          <p className="mt-6 font-mono text-[14px] text-dark-text-secondary leading-relaxed max-w-2xl">
            From gradient descent to GPT — every equation proven,
            every line of code written from zero,
            every research nuance unpacked to the paper.
          </p>

          {/* Stats */}
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
          <div className="mt-12 pt-8 border-t border-dark-border/40 grid grid-cols-3 gap-6 text-[11px] font-mono">
            <ProofPoint icon="∇" label="Math first" sub="every gradient derived" />
            <ProofPoint icon="⌥" label="NumPy → PyTorch" sub="three-layer code pattern" />
            <ProofPoint icon="◈" label="Live widgets" sub="poke it until it breaks" />
          </div>
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
