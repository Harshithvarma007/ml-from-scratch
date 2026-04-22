import { Sigma } from 'lucide-react'
import { cn } from '@/lib/utils'

interface MathBlockProps {
  children: string
  caption?: string
  display?: 'center' | 'left'
}

// Plain monospace math — mirrors the reference blog's ASCII math style.
// No KaTeX: everything we need (x², θᵢ, α, →, ·) renders cleanly in JetBrains Mono.
export default function MathBlock({ children, caption, display = 'left' }: MathBlockProps) {
  return (
    <figure className="term-panel-elevated rounded-lg my-5 overflow-hidden">
      {caption && (
        <div className="px-4 pt-3 flex items-center gap-1.5">
          <Sigma className="w-3 h-3 text-dark-accent" strokeWidth={2} />
          <span className="text-[10px] uppercase tracking-wider font-mono text-dark-text-secondary">
            {caption}
          </span>
        </div>
      )}
      <pre
        className={cn(
          'px-5 py-4 overflow-x-auto font-mono text-[13px] leading-[1.8]',
          'text-dark-text-primary whitespace-pre',
          display === 'center' && 'text-center'
        )}
      >
        {children.replace(/\n$/, '')}
      </pre>
    </figure>
  )
}

// Inline math — use <InlineMath>{'α = 0.1'}</InlineMath> for short expressions.
export function InlineMath({ children }: { children: string }) {
  return (
    <code className="font-mono text-[13px] px-1 py-0.5 bg-dark-surface-elevated border border-dark-border rounded text-dark-text-primary">
      {children}
    </code>
  )
}
