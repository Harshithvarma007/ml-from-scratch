'use client'

import { useCallback, useRef, useState, type KeyboardEvent } from 'react'
import { Loader2, Pencil, Play, RotateCcw, Terminal, X } from 'lucide-react'
import { ensurePackagesForSource, getPyodide, resetPythonGlobals } from '@/lib/pyodide'
import { cn } from '@/lib/utils'
import CopyButton from './CopyButton'

interface Props {
  /** The original, canonical source — what "reset" restores to. */
  code: string
  /**
   * Pre-rendered Shiki HTML for the read-only view. Passed through as a
   * string so we can avoid shipping a syntax highlighter to the client.
   */
  highlightedInnerHtml: string
  /** Optional caption rendered in the header. */
  caption?: string
  /** Optional pre-rendered stdout shown before any run. */
  initialOutput?: string
}

type Status = 'idle' | 'booting' | 'loading-pkg' | 'running' | 'done' | 'error'

// The tab-width of the editable textarea. Matches the Shiki read-only view
// and the house code convention.
const TAB = '    '

// A Python code block the reader can run in-browser. First run on a page
// triggers a one-time Pyodide boot (~6 MB from CDN); subsequent runs reuse
// the warmed runtime and are effectively instant for short snippets.
//
// Pedagogy notes:
// - Start in read-only mode — Shiki-highlighted, beautiful, unambiguous.
// - "Edit" swaps to a plain textarea. No CodeMirror dep; readers are tweaking
//   a few lines, not writing a module. Tab inserts 4 spaces. Cmd/Ctrl+Enter
//   runs.
// - "Reset" restores the original source and clears the output so a chain
//   of experiments doesn't accumulate confusion.
// - stdout streams live: Pyodide's batched stdout callback fires per-print,
//   so a loop that prints each iteration shows progress rather than
//   appearing stuck.
export default function RunnableCodeBlock({
  code,
  highlightedInnerHtml,
  caption,
  initialOutput,
}: Props) {
  const [status, setStatus] = useState<Status>('idle')
  const [output, setOutput] = useState<string>(initialOutput ?? '')
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(code)
  const stdoutBufRef = useRef('')
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  const source = editing ? draft : code
  const isBusy = status === 'booting' || status === 'loading-pkg' || status === 'running'

  const run = useCallback(async () => {
    if (isBusy) return
    stdoutBufRef.current = ''
    setOutput('')
    try {
      setStatus('booting')
      const py = await getPyodide()

      // Route stdout/stderr through our buffer. Pyodide flushes per-print in
      // batched mode, which is what we want for incremental UI updates.
      py.setStdout({
        batched: (s) => {
          stdoutBufRef.current += s
          setOutput(stdoutBufRef.current)
        },
      })
      py.setStderr({
        batched: (s) => {
          stdoutBufRef.current += s
          setOutput(stdoutBufRef.current)
        },
      })

      await ensurePackagesForSource(source, py, () => setStatus('loading-pkg'))

      // Clear globals so a re-run doesn't inherit variables from the previous
      // attempt. Each "run" should feel like a fresh REPL session.
      resetPythonGlobals(py)

      setStatus('running')
      await py.runPythonAsync(source)
      setStatus('done')
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      // Append rather than replace — partial stdout from before the error is
      // often the most useful debugging signal.
      stdoutBufRef.current =
        (stdoutBufRef.current ? stdoutBufRef.current + '\n' : '') + msg
      setOutput(stdoutBufRef.current)
      setStatus('error')
    }
  }, [isBusy, source])

  function reset() {
    if (isBusy) return
    setOutput(initialOutput ?? '')
    setStatus('idle')
    setDraft(code)
    setEditing(false)
  }

  function onTextareaKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    // Cmd/Ctrl+Enter → run. Matches Jupyter / Colab / every REPL ever.
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault()
      run()
      return
    }
    // Tab inserts 4 spaces instead of focusing the next element. Shift+Tab
    // does the native thing so the user isn't trapped.
    if (e.key === 'Tab' && !e.shiftKey) {
      e.preventDefault()
      const el = e.currentTarget
      const { selectionStart, selectionEnd } = el
      const next =
        draft.slice(0, selectionStart) + TAB + draft.slice(selectionEnd)
      setDraft(next)
      // Restore cursor position after React re-renders.
      queueMicrotask(() => {
        if (textareaRef.current) {
          textareaRef.current.selectionStart = selectionStart + TAB.length
          textareaRef.current.selectionEnd = selectionStart + TAB.length
        }
      })
    }
  }

  const statusLabel = {
    idle: null,
    booting: 'booting python…',
    'loading-pkg': 'loading packages…',
    running: 'running…',
    done: 'done',
    error: 'error',
  }[status]

  return (
    <figure className="term-panel rounded-lg overflow-hidden my-5">
      <header className="term-panel-header gap-2 flex-wrap">
        <div className="flex items-center gap-2 min-w-0">
          <Terminal className="w-3 h-3 text-dark-accent" strokeWidth={2} />
          <span className="normal-case tracking-normal text-dark-text-secondary truncate">
            {caption ?? 'python'}
          </span>
          {statusLabel && (
            <span
              className={cn(
                'text-[10px] normal-case tracking-normal font-mono flex items-center gap-1',
                status === 'error' ? 'text-term-rose' : 'text-dark-text-disabled',
              )}
              role="status"
              aria-live="polite"
            >
              {isBusy && <Loader2 className="w-3 h-3 animate-spin" />}
              {statusLabel}
            </span>
          )}
        </div>

        <div className="flex items-center gap-1 shrink-0">
          <HeaderButton
            onClick={() => setEditing((v) => !v)}
            disabled={isBusy}
            label={editing ? 'Cancel edit' : 'Edit code'}
            active={editing}
          >
            {editing ? <X className="w-3 h-3" /> : <Pencil className="w-3 h-3" />}
            <span>{editing ? 'cancel' : 'edit'}</span>
          </HeaderButton>
          <HeaderButton
            onClick={reset}
            disabled={isBusy || (status === 'idle' && !editing && draft === code && output === (initialOutput ?? ''))}
            label="Reset code and output"
          >
            <RotateCcw className="w-3 h-3" />
            <span>reset</span>
          </HeaderButton>
          <HeaderButton
            onClick={run}
            disabled={isBusy}
            label="Run code"
            primary
          >
            <Play className="w-3 h-3" />
            <span>run</span>
          </HeaderButton>
        </div>
      </header>

      <div className="group relative">
        {!editing && <CopyButton code={source} label="Copy python code" />}
        {editing ? (
          <textarea
            ref={textareaRef}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onTextareaKeyDown}
            spellCheck={false}
            aria-label="Editable Python source"
            className={cn(
              'block w-full resize-y min-h-[10rem]',
              'p-4 font-mono text-[12.5px] leading-[1.7]',
              'bg-dark-bg text-dark-text-primary',
              'outline-none focus-visible:ring-2 focus-visible:ring-dark-accent focus-visible:ring-inset',
            )}
          />
        ) : (
          <pre
            className={cn(
              'p-4 overflow-x-auto',
              'font-mono text-[12.5px] leading-[1.7]',
              'text-dark-text-primary',
            )}
            // Shiki output — trusted, generated from our theme tokens.
            dangerouslySetInnerHTML={{ __html: highlightedInnerHtml }}
          />
        )}
      </div>

      {(output || status !== 'idle') && (
        <div
          className={cn(
            'border-t border-dark-border px-4 py-3',
            status === 'error' ? 'bg-term-rose/[0.06]' : 'bg-dark-bg',
          )}
        >
          <div className="flex items-center gap-1.5 mb-1.5">
            <span
              className={cn(
                'text-[10px] uppercase tracking-wider',
                status === 'error' ? 'text-term-rose' : 'text-dark-text-disabled',
              )}
            >
              {status === 'error' ? 'stderr' : 'stdout'}
            </span>
            {status === 'done' && (
              <span className="text-[10px] text-term-green font-mono">✓</span>
            )}
          </div>
          <pre
            className={cn(
              'font-mono text-[12px] leading-[1.65] whitespace-pre overflow-x-auto',
              status === 'error' ? 'text-term-rose' : 'text-dark-text-secondary',
            )}
          >
            {output || ' '}
          </pre>
        </div>
      )}
    </figure>
  )
}

interface HeaderButtonProps {
  onClick: () => void
  disabled?: boolean
  label: string
  primary?: boolean
  active?: boolean
  children: React.ReactNode
}

function HeaderButton({
  onClick,
  disabled,
  label,
  primary,
  active,
  children,
}: HeaderButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      className={cn(
        'inline-flex items-center gap-1 px-1.5 py-0.5 rounded',
        'text-[10px] font-mono uppercase tracking-wider transition-all',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-dark-accent',
        'border',
        primary
          ? 'border-dark-accent/60 bg-dark-accent/10 text-dark-accent hover:bg-dark-accent/20'
          : active
            ? 'border-dark-border-hover bg-dark-surface-elevated text-dark-text-primary'
            : 'border-dark-border bg-dark-surface/60 text-dark-text-disabled hover:text-dark-text-primary hover:border-dark-border-hover',
        disabled && 'opacity-40 cursor-not-allowed hover:text-dark-text-disabled hover:border-dark-border',
      )}
    >
      {children}
    </button>
  )
}
