'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Demonstrate three chat-template formats for an SFT training example. The
// same (user, assistant) pair is rendered as raw plain text, ChatML, and
// Llama-style bracket tokens. Special tokens are highlighted so it's clear
// what the model actually sees. A rough token counter gives a sense of how
// much wrapper overhead each template adds.

type Template = 'raw' | 'chatml' | 'llama'

const TEMPLATES: { key: Template; label: string; note: string }[] = [
  { key: 'raw', label: 'raw', note: 'no markers — the model has no idea where turns begin or end' },
  { key: 'chatml', label: 'ChatML', note: 'OpenAI / Qwen / many 2024 models — explicit role tags' },
  { key: 'llama', label: 'Llama-2 chat', note: 'instruction brackets; simpler, no explicit role names' },
]

// Approximate GPT-style tokenization: split on whitespace + punctuation,
// each chunk ≈ one token. Special markers are single tokens.
function approxTokens(s: string, markers: string[]): number {
  let work = s
  let markerCount = 0
  markers.forEach((m) => {
    const re = new RegExp(m.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')
    const matches = work.match(re)
    if (matches) markerCount += matches.length
    work = work.replace(re, ' ')
  })
  const pieces = work.split(/(\s+|[.,!?;:\-\(\)])/g).filter((p) => p.trim().length > 0)
  return markerCount + Math.ceil(pieces.length * 0.85)
}

function renderWithSpecials(text: string, specials: string[]): { text: string; special: boolean }[] {
  // Split by any of the special markers, preserving them.
  if (!specials.length) return [{ text, special: false }]
  const escaped = specials.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
  const re = new RegExp(`(${escaped.join('|')})`, 'g')
  const pieces = text.split(re).filter((p) => p.length > 0)
  return pieces.map((p) => ({ text: p, special: specials.includes(p) }))
}

export default function SFTDataFormatting() {
  const [user, setUser] = useState('What is the capital of France?')
  const [assistant, setAssistant] = useState('The capital of France is Paris.')
  const [tpl, setTpl] = useState<Template>('chatml')

  const { formatted, specials } = useMemo(() => {
    if (tpl === 'raw') {
      return {
        formatted: `${user}\n${assistant}`,
        specials: [] as string[],
      }
    }
    if (tpl === 'chatml') {
      return {
        formatted: `<|im_start|>user\n${user}<|im_end|>\n<|im_start|>assistant\n${assistant}<|im_end|>`,
        specials: ['<|im_start|>', '<|im_end|>'],
      }
    }
    return {
      formatted: `<s>[INST] ${user} [/INST] ${assistant} </s>`,
      specials: ['<s>', '</s>', '[INST]', '[/INST]'],
    }
  }, [tpl, user, assistant])

  const tokCount = approxTokens(formatted, specials)
  const rawTok = approxTokens(`${user}\n${assistant}`, [])
  const overhead = tokCount - rawTok

  const tplInfo = TEMPLATES.find((t) => t.key === tpl)!

  return (
    <WidgetFrame
      widgetName="SFTDataFormatting"
      label="chat templates — the same pair, three wrappers"
      right={<span className="font-mono">raw · ChatML · Llama-2</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            {TEMPLATES.map((t) => (
              <button
                key={t.key}
                onClick={() => setTpl(t.key)}
                className={cn(
                  'px-2.5 py-1 rounded text-[11px] font-mono transition-all',
                  tpl === t.key
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {t.label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="tokens" value={String(tokCount)} accent="text-term-amber" />
            <Readout label="payload" value={String(rawTok)} accent="text-term-cyan" />
            <Readout label="wrapper overhead" value={`+${overhead}`} accent={overhead > 10 ? 'text-term-rose' : 'text-term-green'} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[300px_1fr] gap-4 overflow-hidden">
        {/* Left: inputs */}
        <div className="flex flex-col gap-3 min-w-0 overflow-hidden">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            user turn
          </div>
          <textarea
            value={user}
            onChange={(e) => setUser(e.target.value)}
            rows={3}
            className="font-mono text-[11px] bg-dark-surface-elevated/40 border border-dark-border rounded p-2 text-dark-text-primary resize-none focus:outline-none focus:border-term-cyan"
          />
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            assistant turn
          </div>
          <textarea
            value={assistant}
            onChange={(e) => setAssistant(e.target.value)}
            rows={3}
            className="font-mono text-[11px] bg-dark-surface-elevated/40 border border-dark-border rounded p-2 text-dark-text-primary resize-none focus:outline-none focus:border-term-amber"
          />
          <div className="text-[10.5px] font-mono text-dark-text-muted leading-snug mt-1">
            <span className="text-term-amber">{tplInfo.label}:</span> {tplInfo.note}
          </div>
        </div>

        {/* Right: formatted output */}
        <div className="flex flex-col gap-2 min-h-0 min-w-0">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            what the tokenizer sees
          </div>
          <pre className="flex-1 min-h-0 font-mono text-[11.5px] leading-relaxed bg-dark-bg border border-dark-border rounded p-3 overflow-auto whitespace-pre-wrap text-dark-text-primary">
            {renderWithSpecials(formatted, specials).map((piece, i) => (
              <span
                key={i}
                className={cn(
                  piece.special && 'text-term-pink bg-term-pink/10 rounded px-0.5',
                )}
              >
                {piece.text}
              </span>
            ))}
          </pre>
          <div className="flex items-center gap-3 pt-1 font-mono text-[10px] text-dark-text-muted">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2 h-2 rounded-sm bg-term-pink" /> special token
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2 h-2 rounded-sm bg-dark-text-primary/70" /> content
            </span>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
