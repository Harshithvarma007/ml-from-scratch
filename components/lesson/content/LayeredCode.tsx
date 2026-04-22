import CodeBlock from './CodeBlock'
import LayeredCodeShell from './LayeredCodeShell'

export interface CodeLayer {
  /** Tab label — e.g. "pure python", "numpy", "pytorch". Lowercase is the house style. */
  label: string
  /** Shiki language — defaults to 'python' since the curriculum is PyTorch-centric. */
  language?: string
  /** The source — passed straight through to <CodeBlock>. */
  code: string
  /** Optional filename-style caption inside the panel header. */
  caption?: string
  /** Optional pre-rendered stdout shown beneath. */
  output?: string
  /** When true AND language is python, renders the Pyodide runnable shell instead of read-only. */
  runnable?: boolean
}

// A tabbed collapse of equivalent code across multiple implementation layers
// — typically the three-layer progression (pure Python → NumPy → PyTorch) that
// appears in most algorithmic lessons. Readers can bounce between tabs
// mid-paragraph without scrolling past three near-identical code panels.
//
// Each layer is rendered through the normal <CodeBlock>, so Shiki highlighting,
// per-tab Pyodide runnability, and stdout all work as before. The shell keeps
// every layer mounted and toggles visibility via the `hidden` attribute —
// so runnable stdout and edit state persist across tab switches.
//
// Usage:
//   <LayeredCode layers={[
//     { label: 'pure python', caption: 'sigmoid_scratch.py', code: '...', runnable: true },
//     { label: 'numpy',       caption: 'sigmoid_np.py',      code: '...', runnable: true },
//     { label: 'pytorch',     caption: 'sigmoid_torch.py',   code: '...' },  // read-only
//   ]} />
export default function LayeredCode({
  layers,
  defaultIndex = 0,
}: {
  layers: CodeLayer[]
  defaultIndex?: number
}) {
  return (
    <LayeredCodeShell labels={layers.map((l) => l.label)} defaultIndex={defaultIndex}>
      {layers.map((l, i) => (
        <CodeBlock
          key={i}
          language={l.language ?? 'python'}
          caption={l.caption}
          output={l.output}
          runnable={l.runnable}
        >
          {l.code}
        </CodeBlock>
      ))}
    </LayeredCodeShell>
  )
}
