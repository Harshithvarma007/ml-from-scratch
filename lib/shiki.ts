import 'server-only'
import { createHighlighter, type Highlighter } from 'shiki'

// Memoized shiki highlighter. Loading grammars + theme is O(100ms) and
// only needs to happen once per server lifetime. The set of languages is
// explicit so we only ship grammars we actually use.
const LANGS = [
  'python',
  'bash',
  'javascript',
  'typescript',
  'tsx',
  'jsx',
  'json',
  'yaml',
  'markdown',
  'text',
] as const

export type SupportedLang = (typeof LANGS)[number]

const THEME = 'github-dark-dimmed' as const

let highlighterPromise: Promise<Highlighter> | null = null

export function getHighlighter(): Promise<Highlighter> {
  if (!highlighterPromise) {
    highlighterPromise = createHighlighter({
      themes: [THEME],
      langs: LANGS as unknown as string[],
    })
  }
  return highlighterPromise
}

// Convert a language alias (or unknown input) to the closest supported Shiki
// grammar. Falls back to plain text for anything we don't recognize so we never
// throw during rendering.
export function normalizeLang(input: string | undefined): SupportedLang {
  if (!input) return 'text'
  const v = input.toLowerCase()
  if (v === 'py' || v === 'python') return 'python'
  if (v === 'sh' || v === 'shell' || v === 'bash') return 'bash'
  if (v === 'js') return 'javascript'
  if (v === 'ts') return 'typescript'
  if (v === 'tsx') return 'tsx'
  if (v === 'jsx') return 'jsx'
  if (v === 'json') return 'json'
  if (v === 'yml' || v === 'yaml') return 'yaml'
  if (v === 'md' || v === 'markdown') return 'markdown'
  return 'text'
}

// Render code to a Shiki-highlighted HTML string. The caller is expected to
// inject this via dangerouslySetInnerHTML inside a <pre>-wrapping container.
// Shiki output is trusted (it's generated from our own theme tokens).
export async function highlightCode(code: string, rawLang?: string): Promise<string> {
  const lang = normalizeLang(rawLang)
  const hl = await getHighlighter()
  return hl.codeToHtml(code, {
    lang,
    theme: THEME,
    // Shiki wraps output in <pre class="shiki ..."><code>...</code></pre>.
    // We strip the outer <pre> in the component and render inside our own.
  })
}
