import type { Difficulty } from '@/lib/roadmap'
import { cn } from '@/lib/utils'

const DIFFICULTY_STYLES: Record<Difficulty, string> = {
  Easy: 'text-term-green border-term-green/30 bg-term-green/5',
  Medium: 'text-term-amber border-term-amber/30 bg-term-amber/5',
  Hard: 'text-term-pink border-term-pink/30 bg-term-pink/5',
}

interface DifficultyPillProps {
  difficulty: Difficulty
  className?: string
}

export default function DifficultyPill({ difficulty, className }: DifficultyPillProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center justify-center',
        'px-1.5 py-[1px] rounded-sm border',
        'font-mono text-[10px] tracking-wide uppercase',
        DIFFICULTY_STYLES[difficulty],
        className
      )}
    >
      {difficulty}
    </span>
  )
}
