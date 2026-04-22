import { Terminal } from 'lucide-react'
import { highlightCode, type SupportedLang } from '@/lib/shiki'
import { cn } from '@/lib/utils'
import CopyButton from './CopyButton'
import RunnableCodeBlock from './RunnableCodeBlock'

// Accept the legacy narrow set as well as every language Shiki is configured
// for. Unknown inputs fall back to plain text inside normalizeLang().
type Language = SupportedLang

interface CodeBlockProps {
  children: string
  language?: Language | (string & {})
  caption?: string
  output?: string // Optional stdout — rendered below, dimmed.
  /**
   * When true AND the block is Python, render the runnable shell that boots
   * Pyodide on click and streams live stdout. Opt-in per block so we don't
   * accidentally make every snippet in the curriculum runnable — some are
   * illustrative fragments that wouldn't execute.
   */
  runnable?: boolean
}

// Strip the outer <pre ...>...</pre> that Shiki emits so we can wrap its
// highlighted <code> in our own terminal-style pre. Shiki's <pre> ships its
// own background color and padding; we want our panel chrome instead.
function stripOuterPre(html: string): string {
  return html
    .replace(/^<pre[^>]*>/, '')
    .replace(/<\/pre>\s*$/, '')
}

// Code block with SSR-precomputed Shiki syntax highlighting.
//
// This is an async Server Component — highlighting happens at build time for
// static lesson routes, so the browser never ships a tokenizer. The only
// client code in this module is the tiny <CopyButton> island for clipboard.
export default async function CodeBlock({
  children,
  language = 'python',
  caption,
  output,
  runnable,
}: CodeBlockProps) {
  const code = children.replace(/\n$/, '')
  const highlightedInner = stripOuterPre(await highlightCode(code, language))

  // Delegate to the client shell when this block is meant to be runnable.
  // We still do SSR highlighting — the server sends the same Shiki HTML to
  // the client, the client just wraps it in run/reset/edit controls.
  if (runnable && (language === 'python' || language === 'py')) {
    return (
      <RunnableCodeBlock
        code={code}
        highlightedInnerHtml={highlightedInner}
        caption={caption}
        initialOutput={output}
      />
    )
  }

  return (
    <figure className="term-panel rounded-lg overflow-hidden my-5">
      {caption && (
        <header className="term-panel-header">
          <div className="flex items-center gap-2">
            <Terminal className="w-3 h-3 text-dark-accent" strokeWidth={2} />
            <span className="normal-case tracking-normal text-dark-text-secondary">
              {caption}
            </span>
          </div>
          <span className="text-[10px] text-dark-text-disabled normal-case tracking-normal">
            {language}
          </span>
        </header>
      )}

      <div className="group relative">
        <CopyButton code={code} label={`Copy ${language} code`} />
        <pre
          className={cn(
            'p-4 overflow-x-auto',
            'font-mono text-[12.5px] leading-[1.7]',
            'text-dark-text-primary',
            !caption && 'pt-4',
          )}
          // Shiki output is HTML generated from our theme tokens — trusted.
          dangerouslySetInnerHTML={{ __html: highlightedInner }}
        />
      </div>

      {output && (
        <div className="border-t border-dark-border bg-dark-bg px-4 py-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <span className="text-[10px] uppercase tracking-wider text-dark-text-disabled">
              stdout
            </span>
          </div>
          <pre className="font-mono text-[12px] leading-[1.65] text-dark-text-secondary whitespace-pre overflow-x-auto">
            {output}
          </pre>
        </div>
      )}
    </figure>
  )
}
