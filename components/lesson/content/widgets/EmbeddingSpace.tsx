'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// A 2D scatter of ~20 hand-picked words arranged into semantic clusters.
// Hover a word to see its three nearest neighbors in cosine space (with lines
// drawn between them). Click a word to pin it as the query — the readout then
// reports cosine similarity of every hovered word against the query.

type Word = {
  word: string
  cluster: 'royalty' | 'animals' | 'motion' | 'food' | 'weather'
  // The displayed 2D position. These are hand-set so clusters read cleanly.
  x: number
  y: number
  // A higher-dim embedding for computing cosine similarity honestly.
  vec: readonly number[]
}

const CLUSTER_COLOR: Record<Word['cluster'], string> = {
  royalty: '#a78bfa',
  animals: '#fbbf24',
  motion: '#67e8f9',
  food: '#f472b6',
  weather: '#4ade80',
}

// The vec field uses a tiny semantic basis so cosine produces believable
// neighbors: [royalty, animal, motion, food, weather, male, female, big, small, human]
const WORDS: Word[] = [
  { word: 'king',    cluster: 'royalty', x: 0.18, y: 0.22, vec: [1.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.6, 0.0, 0.7] },
  { word: 'queen',   cluster: 'royalty', x: 0.28, y: 0.18, vec: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.6, 0.0, 0.7] },
  { word: 'prince',  cluster: 'royalty', x: 0.22, y: 0.32, vec: [0.9, 0.0, 0.0, 0.0, 0.0, 0.85, 0.0, 0.4, 0.3, 0.7] },
  { word: 'throne',  cluster: 'royalty', x: 0.33, y: 0.28, vec: [0.95, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.5, 0.0, 0.3] },

  { word: 'dog',     cluster: 'animals', x: 0.78, y: 0.22, vec: [0.0, 1.0, 0.3, 0.0, 0.0, 0.1, 0.1, 0.3, 0.3, 0.0] },
  { word: 'cat',     cluster: 'animals', x: 0.85, y: 0.30, vec: [0.0, 1.0, 0.2, 0.0, 0.0, 0.1, 0.1, 0.2, 0.5, 0.0] },
  { word: 'puppy',   cluster: 'animals', x: 0.72, y: 0.32, vec: [0.0, 0.95, 0.3, 0.0, 0.0, 0.1, 0.1, 0.1, 0.8, 0.0] },
  { word: 'wolf',    cluster: 'animals', x: 0.80, y: 0.38, vec: [0.0, 0.95, 0.4, 0.0, 0.0, 0.2, 0.0, 0.6, 0.0, 0.0] },

  { word: 'run',     cluster: 'motion',  x: 0.22, y: 0.72, vec: [0.0, 0.1, 1.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.3] },
  { word: 'jump',    cluster: 'motion',  x: 0.30, y: 0.78, vec: [0.0, 0.1, 0.95, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.2] },
  { word: 'swim',    cluster: 'motion',  x: 0.17, y: 0.82, vec: [0.0, 0.1, 0.9, 0.0, 0.1, 0.2, 0.2, 0.0, 0.0, 0.2] },
  { word: 'walk',    cluster: 'motion',  x: 0.26, y: 0.86, vec: [0.0, 0.1, 0.85, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.4] },

  { word: 'bread',   cluster: 'food',    x: 0.55, y: 0.78, vec: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.0] },
  { word: 'apple',   cluster: 'food',    x: 0.62, y: 0.86, vec: [0.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.1, 0.5, 0.0] },
  { word: 'pizza',   cluster: 'food',    x: 0.70, y: 0.74, vec: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0] },
  { word: 'cheese',  cluster: 'food',    x: 0.66, y: 0.82, vec: [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.1, 0.4, 0.0] },

  { word: 'rain',    cluster: 'weather', x: 0.84, y: 0.72, vec: [0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.3, 0.0] },
  { word: 'snow',    cluster: 'weather', x: 0.90, y: 0.80, vec: [0.0, 0.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.4, 0.0] },
  { word: 'storm',   cluster: 'weather', x: 0.80, y: 0.86, vec: [0.0, 0.0, 0.3, 0.0, 0.95, 0.0, 0.0, 0.8, 0.0, 0.0] },
  { word: 'cloud',   cluster: 'weather', x: 0.92, y: 0.68, vec: [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3, 0.3, 0.0] },
]

function cosine(a: readonly number[], b: readonly number[]): number {
  let dot = 0
  let an = 0
  let bn = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    an += a[i] * a[i]
    bn += b[i] * b[i]
  }
  const denom = Math.sqrt(an) * Math.sqrt(bn)
  return denom === 0 ? 0 : dot / denom
}

function neighbors(word: Word, k: number): Word[] {
  return WORDS
    .filter((w) => w.word !== word.word)
    .map((w) => ({ w, s: cosine(word.vec, w.vec) }))
    .sort((a, b) => b.s - a.s)
    .slice(0, k)
    .map((x) => x.w)
}

export default function EmbeddingSpace() {
  const [hover, setHover] = useState<string | null>(null)
  const [query, setQuery] = useState<string>('king')

  const hoveredWord = WORDS.find((w) => w.word === hover) ?? null
  const queryWord = WORDS.find((w) => w.word === query) ?? WORDS[0]
  const hoverNeighbors = useMemo(
    () => (hoveredWord ? neighbors(hoveredWord, 3) : []),
    [hoveredWord],
  )

  const sim = hoveredWord ? cosine(hoveredWord.vec, queryWord.vec) : null

  return (
    <WidgetFrame
      widgetName="EmbeddingSpace"
      label="embedding space — hover to see nearest neighbors"
      right={<span className="font-mono">20 tokens · 10-dim semantic basis · cosine metric</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">query:</span>
            {(['king', 'dog', 'run', 'pizza', 'rain'] as const).map((w) => (
              <button
                key={w}
                onClick={() => setQuery(w)}
                className={cn(
                  'px-2 py-1 rounded text-[10.5px] font-mono uppercase transition-all',
                  query === w
                    ? 'bg-dark-accent text-white'
                    : 'border border-dark-border text-dark-text-secondary hover:text-dark-text-primary',
                )}
              >
                {w}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="hover" value={hover ?? '—'} accent="text-term-amber" />
            <Readout
              label={`cos(${query}, ·)`}
              value={sim === null ? '—' : sim.toFixed(3)}
              accent={sim !== null && sim > 0.6 ? 'text-term-green' : 'text-dark-text-primary'}
            />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <svg viewBox="0 0 1000 600" preserveAspectRatio="none" className="w-full h-full">
          {/* Axis frame */}
          <rect x={20} y={14} width={960} height={572} fill="none" stroke="#1e1e1e" strokeWidth={1} rx={4} />
          {[0.25, 0.5, 0.75].map((f) => (
            <g key={f}>
              <line x1={20 + f * 960} y1={14} x2={20 + f * 960} y2={586} stroke="#151519" strokeWidth={1} />
              <line x1={20} y1={14 + f * 572} x2={980} y2={14 + f * 572} stroke="#151519" strokeWidth={1} />
            </g>
          ))}
          <text x={28} y={28} fontSize="9" fill="#555" fontFamily="JetBrains Mono, monospace">
            PC1 →
          </text>
          <text x={28} y={580} fontSize="9" fill="#555" fontFamily="JetBrains Mono, monospace" transform="rotate(-90 28 580)">
            PC2 →
          </text>

          {/* Cluster labels */}
          {(Object.keys(CLUSTER_COLOR) as Word['cluster'][]).map((c) => {
            const inCluster = WORDS.filter((w) => w.cluster === c)
            const cx = (inCluster.reduce((a, w) => a + w.x, 0) / inCluster.length) * 960 + 20
            const cy = (inCluster.reduce((a, w) => a + w.y, 0) / inCluster.length) * 572 + 14 - 22
            return (
              <text
                key={c}
                x={cx}
                y={cy}
                textAnchor="middle"
                fontSize="9"
                fill={CLUSTER_COLOR[c]}
                fontFamily="JetBrains Mono, monospace"
                opacity={0.55}
              >
                · {c} ·
              </text>
            )
          })}

          {/* Lines to hover neighbors */}
          {hoveredWord &&
            hoverNeighbors.map((n) => (
              <line
                key={n.word}
                x1={hoveredWord.x * 960 + 20}
                y1={hoveredWord.y * 572 + 14}
                x2={n.x * 960 + 20}
                y2={n.y * 572 + 14}
                stroke="#fbbf24"
                strokeWidth={1.2}
                strokeDasharray="4 3"
                opacity={0.65}
              />
            ))}

          {/* Query ring */}
          <circle
            cx={queryWord.x * 960 + 20}
            cy={queryWord.y * 572 + 14}
            r={22}
            fill="none"
            stroke="#4ade80"
            strokeWidth={1.5}
            strokeDasharray="3 3"
            opacity={0.8}
          />

          {/* Points */}
          {WORDS.map((w) => {
            const cx = w.x * 960 + 20
            const cy = w.y * 572 + 14
            const isHover = hover === w.word
            const isQuery = query === w.word
            const isNeighbor = hoverNeighbors.some((n) => n.word === w.word)
            const r = isHover ? 7 : isNeighbor ? 5.5 : 4
            return (
              <g key={w.word} onMouseEnter={() => setHover(w.word)} onMouseLeave={() => setHover(null)} onClick={() => setQuery(w.word)} style={{ cursor: 'pointer' }}>
                <circle
                  cx={cx}
                  cy={cy}
                  r={r}
                  fill={CLUSTER_COLOR[w.cluster]}
                  opacity={isHover || isNeighbor || isQuery ? 1 : 0.7}
                  stroke={isHover ? '#fff' : 'none'}
                  strokeWidth={1.2}
                />
                <text
                  x={cx + 9}
                  y={cy + 4}
                  fontSize="11"
                  fill={isHover ? '#fff' : isNeighbor ? '#fbbf24' : isQuery ? '#4ade80' : '#ccc'}
                  fontFamily="JetBrains Mono, monospace"
                  fontWeight={isHover || isQuery ? 600 : 400}
                >
                  {w.word}
                </text>
                {/* Invisible larger hit target */}
                <circle cx={cx} cy={cy} r={14} fill="transparent" />
              </g>
            )
          })}
        </svg>
      </div>
    </WidgetFrame>
  )
}
