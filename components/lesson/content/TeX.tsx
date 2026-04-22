import katex from 'katex'
import { Sigma } from 'lucide-react'
import { cn } from '@/lib/utils'

// KaTeX-rendered math for cases the monospace MathBlock can't handle well —
// matrices, nested fractions, summations with limits, hand-scripted letters.
// Rendering happens server-side via katex.renderToString (deterministic HTML,
// no hydration mismatch). The KaTeX stylesheet is imported once in globals.css.
//
// Authoring rules:
//   - Use <BlockTex> for standalone equations that earn their own line.
//   - Use <InlineTex> for a short expression inside prose (think \(x^2 + 1\)).
//   - Keep the existing <MathBlock> for ASCII flow diagrams and worked numeric
//     traces where monospace alignment IS the point. KaTeX is for the math
//     that needs typography; MathBlock is for the math that needs a grid.
//
// If KaTeX fails to parse (bad TeX), we fall back to a monospace rendering
// with the raw source so the reader at least sees what was intended and the
// authoring error is visible rather than swallowed.
function renderTex(source: string, displayMode: boolean): string {
  try {
    return katex.renderToString(source, {
      displayMode,
      throwOnError: true,
      strict: 'ignore',
      output: 'html',
      trust: false,
    })
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    if (process.env.NODE_ENV !== 'production') {
      console.warn('[TeX] parse error:', msg, '\nsource:', source)
    }
    return `<code class="text-term-rose">${escape(source)}</code>`
  }
}

function escape(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

interface BlockTexProps {
  children: string
  caption?: string
}

export function BlockTex({ children, caption }: BlockTexProps) {
  const html = renderTex(children, true)
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
      <div
        className={cn('px-5 py-4 overflow-x-auto text-dark-text-primary')}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </figure>
  )
}

export function InlineTex({ children }: { children: string }) {
  const html = renderTex(children, false)
  return (
    <span
      className="text-dark-text-primary align-baseline"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
