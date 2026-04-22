'use client'

import { useEffect, useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { StepForward, SkipBack } from 'lucide-react'

// Animated RLHF training step. Click "step" and a token pulses through
// prompt → policy (LM) → response → reward model → r → KL(policy ‖ ref) → loss → update.
// Each node is clickable; clicking shows its input/output shapes and a
// one-line description. A "run" button auto-animates.

type NodeKey = 'prompt' | 'policy' | 'response' | 'rm' | 'kl' | 'loss' | 'update'

const NODES: {
  key: NodeKey
  x: number
  y: number
  w: number
  h: number
  title: string
  color: string
  inputs: string
  outputs: string
  note: string
}[] = [
  { key: 'prompt', x: 30, y: 40, w: 110, h: 56, title: 'prompt x', color: '#67e8f9', inputs: '—', outputs: 'x: [T_x]', note: 'sampled from prompt dataset' },
  { key: 'policy', x: 180, y: 40, w: 110, h: 56, title: 'policy π_θ', color: '#a78bfa', inputs: 'x', outputs: 'y ~ π_θ(· | x)', note: 'rolls out a response token-by-token' },
  { key: 'response', x: 330, y: 40, w: 110, h: 56, title: 'response y', color: '#4ade80', inputs: 'π_θ samples', outputs: 'y: [T_y]', note: 'the sampled completion' },
  { key: 'rm', x: 480, y: 40, w: 110, h: 56, title: 'reward model', color: '#fbbf24', inputs: '(x, y)', outputs: 'r ∈ ℝ', note: 'frozen, trained from preferences' },
  { key: 'kl', x: 480, y: 150, w: 110, h: 56, title: 'KL(π_θ ‖ π_ref)', color: '#f472b6', inputs: '(x, y)', outputs: 'd: scalar', note: 'how far the policy has drifted' },
  { key: 'loss', x: 330, y: 150, w: 110, h: 56, title: 'loss = r − β·KL', color: '#fb923c', inputs: 'r, d', outputs: 'scalar', note: 'advantage → PPO clipped objective' },
  { key: 'update', x: 180, y: 150, w: 110, h: 56, title: 'policy update', color: '#f87171', inputs: '∇L, θ', outputs: 'θ ← θ + η·∇L', note: 'backprop into π_θ only' },
]

const FLOW: { from: NodeKey; to: NodeKey }[] = [
  { from: 'prompt', to: 'policy' },
  { from: 'policy', to: 'response' },
  { from: 'response', to: 'rm' },
  { from: 'response', to: 'kl' },
  { from: 'rm', to: 'loss' },
  { from: 'kl', to: 'loss' },
  { from: 'loss', to: 'update' },
  { from: 'update', to: 'policy' },
]

function nodeCenter(key: NodeKey): { x: number; y: number } {
  const n = NODES.find((nn) => nn.key === key)!
  return { x: n.x + n.w / 2, y: n.y + n.h / 2 }
}

export default function RLHFPipeline() {
  const [activeFlow, setActiveFlow] = useState(-1) // -1 = none, 0..FLOW.length-1 = that edge
  const [auto, setAuto] = useState(false)
  const [selected, setSelected] = useState<NodeKey>('policy')

  const step = () => setActiveFlow((i) => (i + 1) % FLOW.length)
  const reset = () => { setActiveFlow(-1); setAuto(false) }

  useEffect(() => {
    if (!auto) return
    const id = setInterval(() => setActiveFlow((i) => (i + 1) % FLOW.length), 800)
    return () => clearInterval(id)
  }, [auto])

  // Highlight nodes that are at either end of the active edge
  const activeEdge = activeFlow >= 0 ? FLOW[activeFlow] : null
  const highlightSet = new Set<NodeKey>(
    activeEdge ? [activeEdge.from, activeEdge.to] : [],
  )

  const selectedNode = NODES.find((n) => n.key === selected)!

  return (
    <WidgetFrame
      widgetName="RLHFPipeline"
      label="RLHF pipeline — one training step"
      right={<span className="font-mono">prompt → policy → response → reward − β·KL → update</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <Button onClick={reset} variant="ghost">
              <span className="inline-flex items-center gap-1"><SkipBack size={11} /> reset</span>
            </Button>
            <Button onClick={step} variant="primary" disabled={auto}>
              <span className="inline-flex items-center gap-1">step <StepForward size={11} /></span>
            </Button>
            <Button onClick={() => setAuto(!auto)} variant={auto ? 'primary' : 'ghost'}>
              {auto ? 'pause' : 'run'}
            </Button>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="edge" value={activeFlow < 0 ? '—' : `${activeFlow + 1} / ${FLOW.length}`} accent="text-term-amber" />
            <Readout label="selected" value={selected} accent="text-term-cyan" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 overflow-hidden">
        <svg viewBox="0 0 640 380" className="w-full h-full">
          <defs>
            <marker id="arrow-rlhf-dim" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="#6b7280" />
            </marker>
            <marker id="arrow-rlhf-hot" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="#fbbf24" />
            </marker>
          </defs>

          {/* Edges */}
          {FLOW.map((e, i) => {
            const a = nodeCenter(e.from)
            const b = nodeCenter(e.to)
            // route: if vertical, use straight line; horizontal, straight line
            const isLoop = e.from === 'update' && e.to === 'policy'
            const isActive = activeFlow === i
            return (
              <g key={i}>
                {isLoop ? (
                  <path
                    d={`M ${a.x} ${a.y - 28} C ${a.x} 10, ${b.x} 10, ${b.x} ${b.y - 28}`}
                    fill="none"
                    stroke={isActive ? '#fbbf24' : '#3f3f46'}
                    strokeWidth={isActive ? 2.4 : 1.2}
                    markerEnd={isActive ? 'url(#arrow-rlhf-hot)' : 'url(#arrow-rlhf-dim)'}
                    strokeDasharray={isActive ? undefined : '3 3'}
                  />
                ) : (
                  <line
                    x1={a.x + (b.x > a.x ? 55 : b.x < a.x ? -55 : 0)}
                    y1={a.y + (b.y > a.y ? 28 : b.y < a.y ? -28 : 0)}
                    x2={b.x + (a.x > b.x ? 55 : a.x < b.x ? -55 : 0)}
                    y2={b.y + (a.y > b.y ? 28 : a.y < b.y ? -28 : 0)}
                    stroke={isActive ? '#fbbf24' : '#3f3f46'}
                    strokeWidth={isActive ? 2.4 : 1.2}
                    markerEnd={isActive ? 'url(#arrow-rlhf-hot)' : 'url(#arrow-rlhf-dim)'}
                  />
                )}
                {isActive && (
                  <circle cx={(a.x + b.x) / 2} cy={(a.y + b.y) / 2} r={4} fill="#fbbf24">
                    <animate attributeName="r" values="3;6;3" dur="0.8s" repeatCount="indefinite" />
                  </circle>
                )}
              </g>
            )
          })}

          {/* Nodes */}
          {NODES.map((n) => {
            const isHot = highlightSet.has(n.key)
            const isSel = selected === n.key
            return (
              <g key={n.key} onClick={() => setSelected(n.key)} style={{ cursor: 'pointer' }}>
                <rect
                  x={n.x}
                  y={n.y}
                  width={n.w}
                  height={n.h}
                  rx={8}
                  fill={isHot ? '#1f2937' : '#141420'}
                  stroke={isSel ? n.color : isHot ? n.color : '#3f3f46'}
                  strokeWidth={isSel || isHot ? 2 : 1.2}
                />
                <text
                  x={n.x + n.w / 2}
                  y={n.y + n.h / 2 - 2}
                  textAnchor="middle"
                  fontSize="11"
                  fontFamily="JetBrains Mono, monospace"
                  fill={isSel || isHot ? n.color : '#e5e7eb'}
                >
                  {n.title}
                </text>
                <text
                  x={n.x + n.w / 2}
                  y={n.y + n.h / 2 + 13}
                  textAnchor="middle"
                  fontSize="9"
                  fontFamily="JetBrains Mono, monospace"
                  fill="#888"
                >
                  {n.outputs}
                </text>
              </g>
            )
          })}

          {/* ref model static note */}
          <g>
            <rect x={615 - 30} y={150} width={30} height={56} rx={6} fill="#0f0f1a" stroke="#3f3f46" />
            <text
              x={615 - 15}
              y={182}
              textAnchor="middle"
              fontSize="8"
              fontFamily="JetBrains Mono, monospace"
              fill="#888"
              transform={`rotate(-90, ${615 - 15}, 182)`}
            >
              π_ref (frozen)
            </text>
            <line x1={615 - 30} y1={178} x2={590} y2={178} stroke="#3f3f46" strokeDasharray="2 3" />
          </g>

          {/* Inspection panel */}
          <g transform="translate(30, 240)">
            <rect x={0} y={0} width={580} height={120} rx={6} fill="#0f0f1a" stroke={selectedNode.color} strokeWidth={1.2} />
            <text x={14} y={22} fontSize="11" fontFamily="JetBrains Mono, monospace" fill={selectedNode.color} fontWeight={600}>
              {selectedNode.title}
            </text>
            <text x={14} y={44} fontSize="10.5" fontFamily="JetBrains Mono, monospace" fill="#e5e7eb">
              in:  {selectedNode.inputs}
            </text>
            <text x={14} y={62} fontSize="10.5" fontFamily="JetBrains Mono, monospace" fill="#e5e7eb">
              out: {selectedNode.outputs}
            </text>
            <text x={14} y={86} fontSize="10.5" fontFamily="JetBrains Mono, monospace" fill="#a1a1aa">
              {selectedNode.note}
            </text>
            <text x={14} y={106} fontSize="10" fontFamily="JetBrains Mono, monospace" fill="#666">
              {activeEdge ? `flow: ${activeEdge.from} → ${activeEdge.to}` : 'hit step to pulse the flow · click any node to inspect'}
            </text>
          </g>
        </svg>
      </div>
    </WidgetFrame>
  )
}
