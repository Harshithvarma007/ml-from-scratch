import Hero from '@/components/home/Hero'
import SectionCard from '@/components/home/SectionCard'
import Prompt from '@/components/ui/Prompt'
import { roadmap, totalLessons } from '@/lib/roadmap'

export default function Home() {
  return (
    <div className="min-h-screen bg-dark-bg">
      <Hero />

      {/* Curriculum */}
      <section id="curriculum" className="max-w-5xl mx-auto px-6 lg:px-8 py-16">
        <header className="mb-10 animate-fade-up">
          <div className="flex items-center gap-2 text-[11px] font-mono text-dark-text-muted uppercase tracking-wider">
            <Prompt />
            curriculum · {roadmap.length} sections · {totalLessons()} lessons
          </div>
          <h2 className="mt-3 font-mono text-2xl md:text-3xl text-dark-text-primary tracking-tight">
            The Full Roadmap
          </h2>
          <p className="mt-2 font-sans text-[14px] text-dark-text-secondary max-w-2xl leading-relaxed">
            Fourteen sections, each a self-contained arc. Pick a topic — the section page
            opens with every lesson in order, blurbed and ranked by difficulty.
          </p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {roadmap.map((section, i) => (
            <SectionCard key={section.slug} section={section} index={i} />
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="max-w-5xl mx-auto px-6 lg:px-8 py-10 border-t border-dark-border/60">
        <div className="flex flex-wrap items-center justify-between gap-4 text-[11px] font-mono text-dark-text-disabled">
          <div className="flex items-center gap-2">
            <span className="text-dark-accent">ml</span>
            <span>from·scratch</span>
            <span className="text-dark-text-disabled">·</span>
            <span>v1 · 2026</span>
          </div>
          <div className="flex items-center gap-4">
            <span>learn · derive · ship</span>
          </div>
        </div>
      </footer>
    </div>
  )
}
