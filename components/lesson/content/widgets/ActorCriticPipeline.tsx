'use client'

import { useState } from 'react'
import WidgetFrame from './WidgetFrame'
import { cn } from '@/lib/utils'

// SVG dataflow: a state box feeds the actor (policy π) and the critic (value
// V). The environment returns a reward r and next state s'. The TD error
// δ = r + γV(s') − V(s) branches back: as an advantage into the actor's
// update, and as a regression target into the critic's loss. Clicking any
// labelled node opens its exact update rule. A button toggles between the
// "two separate nets" layout and the "shared backbone" layout.

type NodeKey = 'state' | 'actor' | 'critic' | 'action' | 'env' | 'value' | 'td' | 'backbone'

type NodeInfo = {
  title: string
  body: string
  math: string[]
  color: string
}

const BASE_NODES: Record<NodeKey, NodeInfo> = {
  state: {
    title: 'state s',
    body: 'what the agent observes at time t. feeds both actor and critic.',
    math: ['s_t ∈ S'],
    color: '#67e8f9',
  },
  actor: {
    title: 'actor π_θ(a | s)',
    body: 'outputs a distribution over actions. updated with the policy gradient, scaled by advantage A(s,a).',
    math: ['a_t ~ π_θ(· | s_t)', '∇θ J ≈ ∇θ log π_θ(a_t|s_t) · A(s_t, a_t)'],
    color: '#a78bfa',
  },
  critic: {
    title: 'critic V_φ(s)',
    body: 'estimates expected return from state s. trained to match the bootstrapped target r + γV(s\').',
    math: ['V_φ(s_t) ≈ E[R_t]', 'L_φ = (r_t + γV_φ(s_{t+1}) − V_φ(s_t))²'],
    color: '#fbbf24',
  },
  action: {
    title: 'action a',
    body: 'what the actor picks — sampled from the current policy.',
    math: ['a_t ∈ A'],
    color: '#f472b6',
  },
  env: {
    title: 'environment',
    body: 'executes action a, returns reward r and the next state s\'.',
    math: ['(r_t, s_{t+1}) = env.step(s_t, a_t)'],
    color: '#5eead4',
  },
  value: {
    title: 'V(s)',
    body: 'baseline used to compute advantage. identical output as the critic box — highlighted here because it flows into the TD computation.',
    math: ['V(s_t), V(s_{t+1})'],
    color: '#fbbf24',
  },
  td: {
    title: 'TD error δ',
    body: 'the "surprise": reward plus discounted bootstrap minus our old estimate. drives BOTH updates.',
    math: ['δ_t = r_t + γ V_φ(s_{t+1}) − V_φ(s_t)', 'advantage A(s_t,a_t) ≈ δ_t'],
    color: '#fb7185',
  },
  backbone: {
    title: 'shared backbone',
    body: 'actor and critic reuse the same feature extractor (typical in A2C/A3C/PPO). saves compute and gives the representation a joint job.',
    math: ['h = f_ψ(s)', 'actor head: π(a|s) = softmax(W_π h)', 'critic head: V(s) = W_V h'],
    color: '#c084fc',
  },
}

export default function ActorCriticPipeline() {
  const [sel, setSel] = useState<NodeKey>('td')
  const [shared, setShared] = useState(false)

  const node = BASE_NODES[sel]

  return (
    <WidgetFrame
      widgetName="ActorCriticPipeline"
      label="actor-critic — dataflow and updates"
      right={<span className="font-mono">{shared ? 'shared backbone' : 'two separate nets'}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setShared(false)}
              className={cn(
                'px-2.5 py-1 rounded text-[10.5px] font-mono uppercase tracking-wider transition-all border',
                !shared ? 'border-term-amber text-term-amber bg-term-amber/10' : 'border-dark-border text-dark-text-secondary',
              )}
            >
              two-net
            </button>
            <button
              onClick={() => setShared(true)}
              className={cn(
                'px-2.5 py-1 rounded text-[10.5px] font-mono uppercase tracking-wider transition-all border',
                shared ? 'border-term-amber text-term-amber bg-term-amber/10' : 'border-dark-border text-dark-text-secondary',
              )}
            >
              shared backbone
            </button>
          </div>
          <div className="flex items-center gap-2 ml-auto text-[11px] font-mono text-dark-text-muted">
            click any node for its update rule
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-[1.4fr_1fr] gap-4 overflow-hidden">
        <div className="min-h-0 relative">
          {shared ? (
            <DiagramShared sel={sel} onSelect={setSel} />
          ) : (
            <DiagramSeparate sel={sel} onSelect={setSel} />
          )}
        </div>

        {/* Right: node detail */}
        <div className="flex flex-col gap-3 min-h-0 min-w-0 overflow-auto">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            selected node
          </div>
          <div
            className="rounded border p-4"
            style={{
              borderColor: node.color + '80',
              backgroundColor: node.color + '10',
            }}
          >
            <div className="font-mono text-[13px] mb-2" style={{ color: node.color }}>
              {node.title}
            </div>
            <p className="font-sans text-[12.5px] text-dark-text-primary leading-relaxed">
              {node.body}
            </p>
            <div className="mt-3 pt-3 border-t border-dark-border space-y-1.5 font-mono text-[11.5px] text-dark-text-secondary">
              {node.math.map((m, i) => (
                <div key={i} className="tabular-nums">
                  {m}
                </div>
              ))}
            </div>
          </div>

          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-2">
            flow of δ
          </div>
          <div className="rounded border border-dark-border bg-dark-surface-elevated/30 p-3 font-mono text-[11px] text-dark-text-muted leading-relaxed">
            <span className="text-term-rose">δ = r + γV(s') − V(s)</span> is the single number that updates
            both heads:
            <div className="mt-2 pl-2 space-y-1">
              <div><span className="text-term-purple">actor:</span> scales the log-prob gradient ∇θ log π(a|s)</div>
              <div><span className="text-term-amber">critic:</span> is the regression residual we minimise</div>
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function Node({
  x,
  y,
  w,
  h,
  label,
  sublabel,
  color,
  active,
  onClick,
  dashed,
}: {
  x: number
  y: number
  w: number
  h: number
  label: string
  sublabel?: string
  color: string
  active: boolean
  onClick: () => void
  dashed?: boolean
}) {
  return (
    <g onClick={onClick} style={{ cursor: 'pointer' }}>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={8}
        fill={active ? color + '20' : '#11111a'}
        stroke={color}
        strokeWidth={active ? 2 : 1.2}
        strokeDasharray={dashed ? '4 3' : undefined}
      />
      <text
        x={x + w / 2}
        y={y + h / 2 - (sublabel ? 6 : 0)}
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize="11.5"
        fill={color}
        fontFamily="JetBrains Mono, monospace"
        fontWeight={600}
      >
        {label}
      </text>
      {sublabel && (
        <text
          x={x + w / 2}
          y={y + h / 2 + 12}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize="9.5"
          fill="#888"
          fontFamily="JetBrains Mono, monospace"
        >
          {sublabel}
        </text>
      )}
    </g>
  )
}

function Arrow({
  x1, y1, x2, y2, color = '#3f3f46', label, labelY,
}: {
  x1: number; y1: number; x2: number; y2: number; color?: string; label?: string; labelY?: number
}) {
  return (
    <g>
      <defs>
        <marker
          id={`ac-arrow-${color.replace(/[^a-z0-9]/gi, '')}`}
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M0,0 L10,5 L0,10 z" fill={color} />
        </marker>
      </defs>
      <line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke={color}
        strokeWidth={1.5}
        markerEnd={`url(#ac-arrow-${color.replace(/[^a-z0-9]/gi, '')})`}
      />
      {label && (
        <text
          x={(x1 + x2) / 2}
          y={labelY ?? (y1 + y2) / 2 - 5}
          textAnchor="middle"
          fontSize="10"
          fill={color}
          fontFamily="JetBrains Mono, monospace"
        >
          {label}
        </text>
      )}
    </g>
  )
}

function DiagramSeparate({ sel, onSelect }: { sel: NodeKey; onSelect: (k: NodeKey) => void }) {
  return (
    <svg viewBox="0 0 560 340" className="w-full h-full">
      <Node x={20} y={140} w={100} h={56} label="state s" color="#67e8f9" active={sel === 'state'} onClick={() => onSelect('state')} />
      <Node x={180} y={60} w={120} h={56} label="actor πθ" sublabel="policy net" color="#a78bfa" active={sel === 'actor'} onClick={() => onSelect('actor')} />
      <Node x={180} y={220} w={120} h={56} label="critic Vφ" sublabel="value net" color="#fbbf24" active={sel === 'critic'} onClick={() => onSelect('critic')} />
      <Node x={340} y={60} w={90} h={56} label="action a" color="#f472b6" active={sel === 'action'} onClick={() => onSelect('action')} />
      <Node x={450} y={140} w={90} h={56} label="env" sublabel="r, s'" color="#5eead4" active={sel === 'env'} onClick={() => onSelect('env')} />
      <Node x={340} y={220} w={90} h={56} label="V(s), V(s')" color="#fbbf24" active={sel === 'value'} onClick={() => onSelect('value')} />
      <Node x={200} y={290} w={90} h={36} label="δ = TD error" color="#fb7185" active={sel === 'td'} onClick={() => onSelect('td')} />

      {/* state → actor */}
      <Arrow x1={120} y1={156} x2={180} y2={88} color="#67e8f9" />
      {/* state → critic */}
      <Arrow x1={120} y1={180} x2={180} y2={248} color="#67e8f9" />
      {/* actor → action */}
      <Arrow x1={300} y1={88} x2={340} y2={88} color="#a78bfa" label="a ~ πθ" labelY={82} />
      {/* action → env */}
      <Arrow x1={430} y1={88} x2={475} y2={140} color="#f472b6" />
      {/* env → state (next) */}
      <Arrow x1={495} y1={196} x2={400} y2={248} color="#5eead4" label="r, s'" labelY={228} />
      {/* critic → V */}
      <Arrow x1={300} y1={248} x2={340} y2={248} color="#fbbf24" />
      {/* V → δ */}
      <Arrow x1={340} y1={276} x2={290} y2={306} color="#fb7185" />
      {/* env → δ (reward) */}
      <Arrow x1={450} y1={176} x2={290} y2={306} color="#fb7185" label="r" labelY={230} />
      {/* δ → actor (advantage) */}
      <Arrow x1={245} y1={290} x2={240} y2={116} color="#fb7185" label="A = δ" labelY={180} />
      {/* δ → critic (loss) */}
      <Arrow x1={245} y1={290} x2={240} y2={276} color="#fb7185" />
    </svg>
  )
}

function DiagramShared({ sel, onSelect }: { sel: NodeKey; onSelect: (k: NodeKey) => void }) {
  return (
    <svg viewBox="0 0 560 340" className="w-full h-full">
      <Node x={20} y={140} w={100} h={56} label="state s" color="#67e8f9" active={sel === 'state'} onClick={() => onSelect('state')} />
      <Node x={160} y={130} w={140} h={70} label="backbone fψ" sublabel="shared features h" color="#c084fc" active={sel === 'backbone'} onClick={() => onSelect('backbone')} dashed />
      <Node x={340} y={60} w={120} h={56} label="actor head" sublabel="π(a|s) = softmax(Wπh)" color="#a78bfa" active={sel === 'actor'} onClick={() => onSelect('actor')} />
      <Node x={340} y={220} w={120} h={56} label="critic head" sublabel="V(s) = WV h" color="#fbbf24" active={sel === 'critic'} onClick={() => onSelect('critic')} />
      <Node x={470} y={140} w={70} h={56} label="env" color="#5eead4" active={sel === 'env'} onClick={() => onSelect('env')} />
      <Node x={210} y={290} w={110} h={36} label="δ = TD error" color="#fb7185" active={sel === 'td'} onClick={() => onSelect('td')} />

      <Arrow x1={120} y1={168} x2={160} y2={165} color="#67e8f9" />
      <Arrow x1={300} y1={150} x2={340} y2={88} color="#c084fc" label="h" labelY={112} />
      <Arrow x1={300} y1={175} x2={340} y2={248} color="#c084fc" label="h" labelY={228} />
      <Arrow x1={460} y1={88} x2={495} y2={140} color="#a78bfa" />
      <Arrow x1={495} y1={196} x2={460} y2={248} color="#5eead4" label="r, s'" labelY={238} />
      <Arrow x1={395} y1={276} x2={270} y2={308} color="#fb7185" />
      <Arrow x1={265} y1={290} x2={395} y2={116} color="#fb7185" label="A = δ" labelY={200} />
    </svg>
  )
}
