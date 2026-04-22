// Tiny Python/TypeScript tokenizer for lesson code blocks.
// Hand-rolled to avoid shipping a full highlighter library — the lessons only
// use a narrow subset of Python + occasional plaintext output, so a ~100-line
// regex walker is plenty.

export type TokenType =
  | 'keyword'
  | 'builtin'
  | 'number'
  | 'string'
  | 'comment'
  | 'decorator'
  | 'func'
  | 'self'
  | 'operator'
  | 'punct'
  | 'ident'
  | 'text'

export interface Token {
  type: TokenType
  text: string
}

const PY_KEYWORDS = new Set([
  'def', 'return', 'for', 'in', 'if', 'else', 'elif', 'import', 'from', 'as',
  'while', 'break', 'continue', 'pass', 'class', 'lambda', 'try', 'except',
  'finally', 'with', 'yield', 'and', 'or', 'not', 'is', 'None', 'True', 'False',
  'global', 'nonlocal', 'raise', 'assert', 'del', 'async', 'await',
])

const PY_BUILTINS = new Set([
  'print', 'len', 'zip', 'list', 'dict', 'tuple', 'set', 'enumerate', 'range',
  'map', 'filter', 'sum', 'min', 'max', 'abs', 'round', 'str', 'int', 'float',
  'bool', 'type', 'isinstance', 'open', 'sorted', 'reversed', 'any', 'all',
])

const OPERATORS = new Set([
  '+', '-', '*', '/', '%', '**', '//', '=', '==', '!=', '<', '>', '<=', '>=',
  '+=', '-=', '*=', '/=', '&', '|', '^', '~', '<<', '>>', '->',
])

export function tokenizePython(code: string): Token[] {
  const tokens: Token[] = []
  let i = 0

  while (i < code.length) {
    const rest = code.slice(i)
    const ch = code[i]

    // Comments
    if (ch === '#') {
      const end = code.indexOf('\n', i)
      const text = end === -1 ? rest : code.slice(i, end)
      tokens.push({ type: 'comment', text })
      i += text.length
      continue
    }

    // Strings — single or double quote, no multi-line triple-quote support (we don't use it).
    if (ch === '"' || ch === "'") {
      const quote = ch
      let j = i + 1
      while (j < code.length && code[j] !== quote) {
        if (code[j] === '\\') j++ // skip escaped char
        j++
      }
      const text = code.slice(i, j + 1)
      tokens.push({ type: 'string', text })
      i = j + 1
      continue
    }

    // f-strings — handled same as strings
    if ((ch === 'f' || ch === 'r' || ch === 'b') && (code[i + 1] === '"' || code[i + 1] === "'")) {
      const quote = code[i + 1]
      let j = i + 2
      while (j < code.length && code[j] !== quote) {
        if (code[j] === '\\') j++
        j++
      }
      const text = code.slice(i, j + 1)
      tokens.push({ type: 'string', text })
      i = j + 1
      continue
    }

    // Numbers
    const numMatch = rest.match(/^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/)
    if (numMatch) {
      tokens.push({ type: 'number', text: numMatch[0] })
      i += numMatch[0].length
      continue
    }

    // Decorators
    if (ch === '@') {
      const m = rest.match(/^@[\w.]+/)
      if (m) {
        tokens.push({ type: 'decorator', text: m[0] })
        i += m[0].length
        continue
      }
    }

    // Identifiers — classify as keyword / builtin / function-call / self / plain ident
    const idMatch = rest.match(/^[A-Za-z_]\w*/)
    if (idMatch) {
      const word = idMatch[0]
      let type: TokenType = 'ident'
      if (PY_KEYWORDS.has(word)) type = 'keyword'
      else if (PY_BUILTINS.has(word)) type = 'builtin'
      else if (word === 'self' || word === 'cls') type = 'self'
      else {
        // Look ahead for `(` to mark function calls.
        const next = code.slice(i + word.length).match(/^\s*\(/)
        if (next) type = 'func'
      }
      tokens.push({ type, text: word })
      i += word.length
      continue
    }

    // Multi-char operators
    const op3 = rest.slice(0, 3)
    const op2 = rest.slice(0, 2)
    if (OPERATORS.has(op3)) {
      tokens.push({ type: 'operator', text: op3 })
      i += 3
      continue
    }
    if (OPERATORS.has(op2)) {
      tokens.push({ type: 'operator', text: op2 })
      i += 2
      continue
    }
    if (OPERATORS.has(ch)) {
      tokens.push({ type: 'operator', text: ch })
      i++
      continue
    }

    // Whitespace / punctuation / anything else — pass through as plain text.
    tokens.push({ type: 'text', text: ch })
    i++
  }

  return tokens
}

// Tailwind class per token type. Tuned against the terminal palette.
export const tokenClass: Record<TokenType, string> = {
  keyword: 'text-term-purple',
  builtin: 'text-term-cyan',
  number: 'text-term-orange',
  string: 'text-term-green',
  comment: 'text-dark-text-muted italic',
  decorator: 'text-term-amber',
  func: 'text-term-cyan',
  self: 'text-term-rose',
  operator: 'text-dark-text-secondary',
  punct: 'text-dark-text-secondary',
  ident: 'text-dark-text-primary',
  text: '',
}
