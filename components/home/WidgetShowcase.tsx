'use client'

import dynamic from 'next/dynamic'
import Link from 'next/link'
import { useEffect, useRef, useState } from 'react'
import { ArrowRight } from 'lucide-react'
import { cn } from '@/lib/utils'

const AttentionHeatmap = dynamic(
  () => import('@/components/lesson/content/widgets/AttentionHeatmap'),
  { ssr: false }
)

export default function WidgetShowcase() {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setVisible(true); observer.disconnect() } },
      { threshold: 0.15 }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  return (
    <section className="relative py-24 overflow-hidden">
      {/* Ambient glow */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[500px] bg-dark-accent/5 rounded-full blur-3xl" />
      </div>

      {/* Top divider */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-dark-border to-transparent" />

      <div
        ref={ref}
        className={cn(
          'max-w-4xl mx-auto px-6 lg:px-8 transition-all duration-700 ease-out',
          visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
        )}
      >
        {/* Label */}
        <div className="flex items-center justify-center gap-3 mb-10">
          <div className="h-px flex-1 bg-gradient-to-r from-transparent to-dark-border/60" />
          <span className="font-mono text-[11px] text-dark-text-muted uppercase tracking-widest px-2">
            this is what a lesson feels like
          </span>
          <div className="h-px flex-1 bg-gradient-to-l from-transparent to-dark-border/60" />
        </div>

        {/* Widget — chromeless */}
        <div className="relative rounded-xl overflow-hidden ring-1 ring-dark-border/40 shadow-dark-lg">
          {/* Accent top line */}
          <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-dark-accent/50 to-transparent z-10" />
          <AttentionHeatmap />
        </div>

        {/* CTA */}
        <div className="mt-10 flex flex-col items-center gap-3">
          <p className="font-mono text-[12px] text-dark-text-muted">
            hover the rows · toggle causal masking · drag the temperature
          </p>
          <Link
            href="/learn/attention-and-transformers/self-attention"
            className="group inline-flex items-center gap-2 font-mono text-[13px] text-dark-accent hover:text-dark-accent-hover transition-colors"
          >
            See the full derivation — Self Attention
            <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-0.5" />
          </Link>
          <p className="font-mono text-[11px] text-dark-text-disabled">
            168 more widgets inside
          </p>
        </div>
      </div>

      {/* Bottom divider */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-dark-border to-transparent" />
    </section>
  )
}
