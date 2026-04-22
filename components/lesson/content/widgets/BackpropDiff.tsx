'use client'

import { useCallback, useRef, useState, type KeyboardEvent } from 'react'
import { Check, Loader2, Play, RotateCcw, Terminal, X as XIcon } from 'lucide-react'
import { ensurePackagesForSource, getPyodide, resetPythonGlobals } from '@/lib/pyodide'
import { cn } from '@/lib/utils'

// Reader's starter — the backward() they're asked to fill in. A deliberately
// wrong identity-returning stub so every gradient will fail on first run and
// the reader gets a clean "now go fix them, one at a time" signal.
const STARTER_BACKWARD = `import numpy as np

# Fill in the real gradients. Cache contains (X, z1, h, z2, logp) from forward.
# y is the integer class labels.
def backward(X, y, W1, b1, W2, b2, cache):
    X_, z1, h, z2, logp = cache
    N = X_.shape[0]

    # ---- YOUR CODE HERE ----
    # Hints:
    #   p   = np.exp(logp)
    #   dz2 = (p - one_hot(y, K)) / N          # softmax + NLL combined
    #   dW2 = h.T @ dz2;   db2 = dz2.sum(0)
    #   dh  = dz2 @ W2.T
    #   dz1 = dh * (1 - h**2)                   # tanh' = 1 - tanh^2
    #   dW1 = X_.T @ dz1;  db1 = dz1.sum(0)
    dW1 = np.zeros_like(W1)
    db1 = np.zeros_like(b1)
    dW2 = np.zeros_like(W2)
    db2 = np.zeros_like(b2)
    # ---- END YOUR CODE ----

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
`

// Harness that surrounds the reader's code. Builds a tiny reproducible MLP,
// calls their backward(), computes a central-difference numerical gradient
// for every parameter, and emits a sentinel-tagged JSON report we can parse
// out of Pyodide's stdout.
const SETUP_PY = `import numpy as np
import json as _json

_rng = np.random.default_rng(0)
N, D_in, D_hid, D_out = 4, 3, 5, 3
X  = _rng.standard_normal((N, D_in))
y  = _rng.integers(0, D_out, size=N)
W1 = _rng.standard_normal((D_in, D_hid)) * 0.5
b1 = _rng.standard_normal(D_hid) * 0.1
W2 = _rng.standard_normal((D_hid, D_out)) * 0.5
b2 = _rng.standard_normal(D_out) * 0.1

def _forward(X, W1, b1, W2, b2, y):
    z1 = X @ W1 + b1
    h  = np.tanh(z1)
    z2 = h @ W2 + b2
    z2s = z2 - z2.max(axis=1, keepdims=True)
    logp = z2s - np.log(np.exp(z2s).sum(axis=1, keepdims=True))
    loss = -logp[np.arange(len(y)), y].mean()
    return loss, (X, z1, h, z2, logp)
`

const CHECK_PY = `
_loss, _cache = _forward(X, W1, b1, W2, b2, y)
try:
    _grads = backward(X, y, W1, b1, W2, b2, _cache)
except Exception as _e:
    print("@@BACKPROP_DIFF@@" + _json.dumps({"pyerror": f"{type(_e).__name__}: {_e}"}))
    raise SystemExit(0)

def _numeric_grad(P):
    eps = 1e-5
    g = np.zeros_like(P, dtype=float)
    flat = P.reshape(-1)
    gflat = g.reshape(-1)
    for i in range(flat.size):
        orig = float(flat[i])
        flat[i] = orig + eps; a = _forward(X, W1, b1, W2, b2, y)[0]
        flat[i] = orig - eps; b = _forward(X, W1, b1, W2, b2, y)[0]
        flat[i] = orig
        gflat[i] = (a - b) / (2 * eps)
    return g

_report = {}
for _name, _P in (("dW1", W1), ("db1", b1), ("dW2", W2), ("db2", b2)):
    _num = _numeric_grad(_P)
    _ana = _grads.get(_name)
    _entry = {"shape": list(_num.shape)}
    if _ana is None:
        _entry["status"] = "missing"
    else:
        _ana = np.asarray(_ana, dtype=float)
        if _ana.shape != _num.shape:
            _entry["status"] = "shape"
            _entry["got_shape"] = list(_ana.shape)
        else:
            _err = float(np.abs(_ana - _num).max())
            _entry["status"] = "pass" if _err < 1e-4 else "fail"
            _entry["error"] = _err
    _report[_name] = _entry

print("@@BACKPROP_DIFF@@" + _json.dumps({"report": _report, "loss": float(_loss)}))
`

type RowStatus = 'pass' | 'fail' | 'missing' | 'shape'

interface Row {
  name: string
  status: RowStatus
  shape: number[]
  error?: number
  got_shape?: number[]
}

interface ReportState {
  rows: Row[]
  loss: number
}

type Status = 'idle' | 'booting' | 'loading-pkg' | 'running' | 'done' | 'error'

const TAB = '    '
const ROWS_ORDER = ['dW1', 'db1', 'dW2', 'db2'] as const

// Karpathy-style "becoming a backprop ninja" exercise in-browser. The reader
// fills in a backward() pass for a tiny 2-layer MLP with tanh + softmax; the
// widget runs it alongside a finite-difference ground truth and reports
// max-abs-error per tensor. Four green check-marks = ninja achieved.
export default function BackpropDiff() {
  const [code, setCode] = useState(STARTER_BACKWARD)
  const [status, setStatus] = useState<Status>('idle')
  const [errorText, setErrorText] = useState<string | null>(null)
  const [report, setReport] = useState<ReportState | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  const isBusy = status === 'booting' || status === 'loading-pkg' || status === 'running'

  const reset = useCallback(() => {
    if (isBusy) return
    setCode(STARTER_BACKWARD)
    setReport(null)
    setErrorText(null)
    setStatus('idle')
  }, [isBusy])

  const run = useCallback(async () => {
    if (isBusy) return
    setErrorText(null)
    setReport(null)
    const source = `${SETUP_PY}\n${code}\n${CHECK_PY}`
    let stdoutBuf = ''
    try {
      setStatus('booting')
      const py = await getPyodide()
      py.setStdout({
        batched: (s) => {
          stdoutBuf += s
        },
      })
      py.setStderr({
        batched: (s) => {
          stdoutBuf += s
        },
      })
      await ensurePackagesForSource(source, py, () => setStatus('loading-pkg'))
      resetPythonGlobals(py)
      setStatus('running')
      await py.runPythonAsync(source)

      // Parse the sentinel line emitted by CHECK_PY.
      const marker = '@@BACKPROP_DIFF@@'
      const line = stdoutBuf
        .split('\n')
        .map((l) => l.trim())
        .find((l) => l.startsWith(marker))
      if (!line) {
        throw new Error('No gradient report emitted. Did `backward()` return a dict?')
      }
      const payload = JSON.parse(line.slice(marker.length)) as
        | { pyerror: string }
        | { report: Record<string, Omit<Row, 'name'>>; loss: number }

      if ('pyerror' in payload) {
        setErrorText(payload.pyerror)
        setStatus('error')
        return
      }

      const rows: Row[] = ROWS_ORDER.map((name) => ({
        name,
        ...payload.report[name],
      }))
      setReport({ rows, loss: payload.loss })
      setStatus('done')
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setErrorText(
        stdoutBuf
          ? `${stdoutBuf.trim()}\n${msg}`
          : msg,
      )
      setStatus('error')
    }
  }, [code, isBusy])

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault()
      run()
      return
    }
    if (e.key === 'Tab' && !e.shiftKey) {
      e.preventDefault()
      const el = e.currentTarget
      const { selectionStart, selectionEnd } = el
      const next = code.slice(0, selectionStart) + TAB + code.slice(selectionEnd)
      setCode(next)
      queueMicrotask(() => {
        if (textareaRef.current) {
          textareaRef.current.selectionStart = selectionStart + TAB.length
          textareaRef.current.selectionEnd = selectionStart + TAB.length
        }
      })
    }
  }

  const passCount = report?.rows.filter((r) => r.status === 'pass').length ?? 0
  const total = ROWS_ORDER.length
  const allPass = report && passCount === total

  const statusLabel = {
    idle: null,
    booting: 'booting python…',
    'loading-pkg': 'loading numpy…',
    running: 'checking gradients…',
    done: null,
    error: null,
  }[status]

  return (
    <figure
      className="term-panel rounded-lg overflow-hidden my-6"
      role="region"
      aria-label="Backprop Ninja — check your analytic gradients against finite differences"
    >
      <header className="term-panel-header gap-2 flex-wrap">
        <div className="flex items-center gap-2 min-w-0">
          <Terminal className="w-3 h-3 text-dark-accent" strokeWidth={2} />
          <span className="normal-case tracking-normal text-dark-text-secondary">
            backprop_ninja.py · fill in <code className="font-mono">backward()</code>
          </span>
          {statusLabel && (
            <span
              className="text-[10px] normal-case tracking-normal font-mono text-dark-text-disabled flex items-center gap-1"
              role="status"
              aria-live="polite"
            >
              {isBusy && <Loader2 className="w-3 h-3 animate-spin" />}
              {statusLabel}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <HeaderButton onClick={reset} disabled={isBusy} label="Reset code">
            <RotateCcw className="w-3 h-3" />
            <span>reset</span>
          </HeaderButton>
          <HeaderButton
            onClick={run}
            disabled={isBusy}
            label="Run gradient check"
            primary
          >
            <Play className="w-3 h-3" />
            <span>check</span>
          </HeaderButton>
        </div>
      </header>

      <textarea
        ref={textareaRef}
        value={code}
        onChange={(e) => setCode(e.target.value)}
        onKeyDown={onKeyDown}
        spellCheck={false}
        aria-label="Editable Python backward pass"
        className={cn(
          'block w-full resize-y min-h-[18rem]',
          'p-4 font-mono text-[12.5px] leading-[1.7]',
          'bg-dark-bg text-dark-text-primary',
          'outline-none focus-visible:ring-2 focus-visible:ring-dark-accent focus-visible:ring-inset',
        )}
      />

      {/* Report */}
      {(report || errorText) && (
        <div
          className={cn(
            'border-t border-dark-border px-4 py-4',
            errorText ? 'bg-term-rose/[0.06]' : 'bg-dark-bg',
          )}
        >
          {errorText ? (
            <>
              <div className="flex items-center gap-1.5 mb-1.5">
                <span className="text-[10px] uppercase tracking-wider text-term-rose">
                  python error
                </span>
              </div>
              <pre className="font-mono text-[12px] leading-[1.65] text-term-rose whitespace-pre-wrap">
                {errorText}
              </pre>
            </>
          ) : report ? (
            <>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span
                    className={cn(
                      'text-[10px] uppercase tracking-wider font-mono',
                      allPass ? 'text-term-green' : 'text-dark-text-disabled',
                    )}
                  >
                    gradient check
                  </span>
                  <span className="text-[11px] font-mono text-dark-text-secondary tabular-nums">
                    {passCount}/{total} passing
                  </span>
                  {allPass && (
                    <span className="text-[11px] font-mono text-term-green">
                      ✓ ninja
                    </span>
                  )}
                </div>
                <span className="text-[10px] font-mono text-dark-text-disabled tabular-nums">
                  loss = {report.loss.toFixed(4)}
                </span>
              </div>
              <ul className="space-y-1.5">
                {report.rows.map((row) => (
                  <GradRow key={row.name} row={row} />
                ))}
              </ul>
              <p className="mt-3 text-[11px] font-mono text-dark-text-disabled leading-relaxed">
                <span className="text-dark-text-muted">pass</span> = max|analytic −
                numeric| &lt; 1e-4. Finite difference is central, ε = 1e-5.
              </p>
            </>
          ) : null}
        </div>
      )}
    </figure>
  )
}

function GradRow({ row }: { row: Row }) {
  const shapeStr = `(${row.shape.join(', ')})`
  const isPass = row.status === 'pass'
  const isFail = row.status === 'fail'
  const isMissing = row.status === 'missing'
  const isShape = row.status === 'shape'

  return (
    <li
      className={cn(
        'grid grid-cols-[24px_1fr_auto_auto] items-center gap-3',
        'font-mono text-[12px] tabular-nums',
        'px-2 py-1 rounded border',
        isPass && 'border-term-green/30 bg-term-green/[0.04]',
        isFail && 'border-term-rose/30 bg-term-rose/[0.04]',
        (isMissing || isShape) && 'border-term-amber/30 bg-term-amber/[0.04]',
      )}
    >
      <span
        aria-hidden
        className={cn(
          'inline-flex items-center justify-center w-5 h-5 rounded',
          isPass && 'text-term-green',
          isFail && 'text-term-rose',
          (isMissing || isShape) && 'text-term-amber',
        )}
      >
        {isPass ? <Check className="w-3.5 h-3.5" strokeWidth={2.5} /> : <XIcon className="w-3.5 h-3.5" strokeWidth={2.5} />}
      </span>
      <span className="text-dark-text-primary">{row.name}</span>
      <span className="text-dark-text-disabled">{shapeStr}</span>
      <span
        className={cn(
          'text-[11px]',
          isPass && 'text-term-green',
          isFail && 'text-term-rose',
          (isMissing || isShape) && 'text-term-amber',
        )}
      >
        {isPass && `max diff ${row.error!.toExponential(2)}`}
        {isFail && `max diff ${row.error!.toExponential(2)}`}
        {isMissing && 'missing from returned dict'}
        {isShape && `wrong shape — got (${row.got_shape!.join(', ')})`}
      </span>
    </li>
  )
}

interface HeaderButtonProps {
  onClick: () => void
  disabled?: boolean
  label: string
  primary?: boolean
  children: React.ReactNode
}

function HeaderButton({
  onClick,
  disabled,
  label,
  primary,
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
          ? 'border-term-green/60 bg-term-green/10 text-term-green hover:bg-term-green/20'
          : 'border-dark-border bg-dark-surface/60 text-dark-text-disabled hover:text-dark-text-primary hover:border-dark-border-hover',
        disabled && 'opacity-40 cursor-not-allowed hover:text-dark-text-disabled hover:border-dark-border',
      )}
    >
      {children}
    </button>
  )
}
