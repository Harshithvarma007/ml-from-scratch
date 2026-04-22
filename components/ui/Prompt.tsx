import { cn } from '@/lib/utils'

interface PromptProps {
  text?: string
  className?: string
}

// Terminal prompt prefix, Octa-style: purple "ml ›" string.
export default function Prompt({ text = 'ml ›', className }: PromptProps) {
  return (
    <span className={cn('term-prompt font-mono text-[11px] tracking-wide', className)}>
      {text}
    </span>
  )
}
