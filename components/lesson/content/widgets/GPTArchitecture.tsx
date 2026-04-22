'use client'

import { useMemo, useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Full GPT architecture as a click-to-inspect SVG stack. Every block exposes
// its parameter count, per-token FLOPs, and a short description. An N slider
// (2..48) rescales the transformer block stack and updates total params.

type PartKey = 'embed' | 'pos' | 'block' | 'ln_f' | 'lm_head'

const D_MODEL = 768
const N_HEADS = 12
const VOCAB = 50257
const CTX = 1024
const FF_MULT = 4

function blockParams(d: number, ff: number): number {
  // attention: QKV (3 * d*d + 3d bias) + out (d*d + d bias) = 4d^2 + 4d
  const attn = 4 * d * d + 4 * d
  // MLP: up (d*ff + ff) + down (ff*d + d)
  const mlp = d * ff + ff + ff * d + d
  // 2 layer norms: 2 * (2d)
  const ln = 4 * d
  return attn + mlp + ln
}

function embedParams(vocab: number, d: number, ctx: number): number {
  return vocab * d + ctx * d
}

function lmHeadParams(vocab: number, d: number): number {
  // GPT-2 ties to embedding, so incremental params are 0. We report the
  // logical matrix size anyway.
  return vocab * d
}

function formatBig(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`
  return String(n)
}

type PartInfo = {
  key: PartKey
  name: string
  color: string
  params: number
  flops: string
  desc: string
}

export default function GPTArchitecture() {
  const [n, setN] = useState(12)
  const [selected, setSelected] = useState<PartKey>('block')

  const parts = useMemo<PartInfo[]>(() => {
    const ff = FF_MULT * D_MODEL
    const bp = blockParams(D_MODEL, ff)
    return [
      {
        key: 'embed',
        name: 'token embedding',
        color: '#67e8f9',
        params: VOCAB * D_MODEL,
        flops: 'O(1) lookup',
        desc: `lookup table of shape (${VOCAB}, ${D_MODEL}). each input id picks one row. this is the only way integers enter the network.`,
      },
      {
        key: 'pos',
        name: 'positional embedding',
        color: '#a78bfa',
        params: CTX * D_MODEL,
        flops: 'O(1) lookup',
        desc: `lookup table of shape (${CTX}, ${D_MODEL}). position t pulls row t, which gets added to the token embedding so the block stack knows order.`,
      },
      {
        key: 'block',
        name: `${n} × transformer block`,
        color: '#fbbf24',
        params: n * bp,
        flops: `~${formatBig(2 * n * bp)} FLOPs/token`,
        desc: `each block: multi-head attention (4d² + 4d weights) + MLP (8d² + 5d weights) + 2 layer norms. ${n} stacked, residual-and-norm between. this is where the thinking happens.`,
      },
      {
        key: 'ln_f',
        name: 'final layer norm',
        color: '#f472b6',
        params: 2 * D_MODEL,
        flops: `${formatBig(5 * D_MODEL)} FLOPs/token`,
        desc: `one last normalization of the residual stream before we project to vocabulary. γ and β vectors of length ${D_MODEL}.`,
      },
      {
        key: 'lm_head',
        name: 'lm_head (tied)',
        color: '#4ade80',
        params: lmHeadParams(VOCAB, D_MODEL),
        flops: `${formatBig(2 * VOCAB * D_MODEL)} FLOPs/token`,
        desc: `projects the ${D_MODEL}-dim final residual to a ${VOCAB}-dim logit vector. weight is tied to the input embedding — no extra params, but the FLOPs still happen every step.`,
      },
    ]
  }, [n])

  const totalParams = embedParams(VOCAB, D_MODEL, CTX) + n * blockParams(D_MODEL, FF_MULT * D_MODEL) + 2 * D_MODEL

  const active = parts.find((p) => p.key === selected) ?? parts[2]

  return (
    <WidgetFrame
      widgetName="GPTArchitecture"
      label="GPT top-down architecture — click any block to inspect"
      right={<span className="font-mono">d_model={D_MODEL} · n_heads={N_HEADS} · vocab={VOCAB} · ctx={CTX}</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="N blocks"
            value={n}
            min={2}
            max={48}
            step={1}
            onChange={(v) => setN(Math.round(v))}
            format={(v) => String(Math.round(v))}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="total params" value={formatBig(totalParams)} accent="text-term-amber" />
            <Readout label="per-token compute" value={formatBig(2 * totalParams)} accent="text-term-green" />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 grid grid-cols-1 md:grid-cols-[300px_1fr] gap-4 overflow-hidden">
        {/* Left: architecture SVG */}
        <ArchSVG n={n} selected={selected} onSelect={setSelected} />

        {/* Right: inspector panel */}
        <div className="flex flex-col gap-3 min-w-0 overflow-auto">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            inspector
          </div>
          <div className="rounded border border-dark-border bg-dark-surface-elevated/40 p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-[12px]" style={{ color: active.color }}>
                {active.name}
              </span>
              <span className="font-mono text-[10px] text-dark-text-disabled tabular-nums">
                {((active.params / totalParams) * 100).toFixed(1)}% of model
              </span>
            </div>
            <div className="font-mono text-[11px] leading-relaxed text-dark-text-muted">
              {active.desc}
            </div>
            <div className="border-t border-dark-border/60 mt-3 pt-2 flex items-center gap-4 font-mono text-[10.5px]">
              <span>
                <span className="text-dark-text-disabled">params:</span>{' '}
                <span className="text-term-amber tabular-nums">{formatBig(active.params)}</span>
              </span>
              <span>
                <span className="text-dark-text-disabled">compute:</span>{' '}
                <span className="text-term-green tabular-nums">{active.flops}</span>
              </span>
            </div>
          </div>

          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled mt-2">
            param breakdown (current N = {n})
          </div>
          <div className="flex flex-col gap-1.5">
            {parts.map((p) => (
              <button
                key={p.key}
                onClick={() => setSelected(p.key)}
                className={cn(
                  'flex items-center gap-2 font-mono text-[10.5px] text-left px-2 py-1 rounded transition-all',
                  selected === p.key
                    ? 'bg-dark-surface-elevated/60 outline outline-1 outline-dark-border-hover'
                    : 'hover:bg-dark-surface-elevated/30',
                )}
              >
                <span className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ backgroundColor: p.color }} />
                <span className="flex-1 text-dark-text-secondary truncate">{p.name}</span>
                <div className="flex-1 h-1.5 bg-dark-surface-elevated/60 rounded-full overflow-hidden max-w-[90px]">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${(p.params / totalParams) * 100}%`,
                      backgroundColor: p.color,
                      opacity: 0.85,
                    }}
                  />
                </div>
                <span className="tabular-nums w-14 text-right text-dark-text-primary">
                  {formatBig(p.params)}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}

function ArchSVG({
  n,
  selected,
  onSelect,
}: {
  n: number
  selected: PartKey
  onSelect: (k: PartKey) => void
}) {
  // Fixed vertical layout — the block stack gets its own internal scaled region.
  const W = 280
  const H = 520
  const cx = W / 2
  const boxW = 200
  const boxH = 34
  const gap = 10

  // Stack row heights
  const embedY = 36
  const posY = embedY + boxH + gap
  const blocksTop = posY + boxH + gap + 18
  const blocksBottom = H - 36 - boxH - gap - 18 - boxH - 12
  const blocksH = blocksBottom - blocksTop
  const lnY = blocksBottom + 18
  const lmY = lnY + boxH + gap

  const maxDraw = Math.min(n, 14)
  const subBoxH = Math.max(5, (blocksH - (maxDraw - 1) * 2) / maxDraw)

  return (
    <div className="relative w-full h-full min-h-0">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full" preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id="blockGradient" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#fbbf24" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.6" />
          </linearGradient>
          <marker id="gpt-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="#525252" />
          </marker>
        </defs>

        {/* Token input pill */}
        <text x={cx} y={16} textAnchor="middle" fontSize="10" fill="#a1a1aa" fontFamily="JetBrains Mono, monospace">
          ids in: [15, 9025, 11, ...]
        </text>
        <line x1={cx} y1={22} x2={cx} y2={embedY} stroke="#525252" strokeWidth={1} markerEnd="url(#gpt-arrow)" />

        <ArchBlock
          cx={cx}
          y={embedY}
          w={boxW}
          h={boxH}
          color="#67e8f9"
          label="token embedding"
          sublabel={`(${VOCAB}, ${D_MODEL})`}
          active={selected === 'embed'}
          onClick={() => onSelect('embed')}
        />
        <line x1={cx} y1={embedY + boxH} x2={cx} y2={posY} stroke="#525252" strokeWidth={1} />

        <ArchBlock
          cx={cx}
          y={posY}
          w={boxW}
          h={boxH}
          color="#a78bfa"
          label="+ positional embedding"
          sublabel={`(${CTX}, ${D_MODEL})`}
          active={selected === 'pos'}
          onClick={() => onSelect('pos')}
        />
        <line x1={cx} y1={posY + boxH} x2={cx} y2={blocksTop - 6} stroke="#525252" strokeWidth={1} markerEnd="url(#gpt-arrow)" />

        {/* Block stack frame */}
        <rect
          x={cx - boxW / 2 - 4}
          y={blocksTop - 4}
          width={boxW + 8}
          height={blocksH + 8}
          rx={8}
          fill="none"
          stroke={selected === 'block' ? '#fbbf24' : '#3f3f46'}
          strokeDasharray="4 4"
          strokeWidth={selected === 'block' ? 1.6 : 1}
          onClick={() => onSelect('block')}
          style={{ cursor: 'pointer' }}
        />
        <text
          x={cx - boxW / 2 + 6}
          y={blocksTop + 11}
          fontSize="9"
          fill={selected === 'block' ? '#fbbf24' : '#71717a'}
          fontFamily="JetBrains Mono, monospace"
        >
          N = {n}
        </text>
        {Array.from({ length: maxDraw }).map((_, k) => {
          const y = blocksTop + 16 + k * (subBoxH + 2)
          return (
            <g key={k} onClick={() => onSelect('block')} style={{ cursor: 'pointer' }}>
              <rect
                x={cx - boxW / 2 + 14}
                y={y}
                width={boxW - 28}
                height={subBoxH}
                rx={2}
                fill="url(#blockGradient)"
                fillOpacity={selected === 'block' ? 0.95 : 0.75}
                stroke={selected === 'block' ? '#fbbf24' : 'transparent'}
                strokeWidth={0.7}
              />
            </g>
          )
        })}
        {n > maxDraw && (
          <text x={cx} y={blocksBottom - 2} textAnchor="middle" fontSize="9" fill="#71717a" fontFamily="JetBrains Mono, monospace">
            ...showing {maxDraw} of {n}
          </text>
        )}

        <line x1={cx} y1={blocksBottom + 4} x2={cx} y2={lnY} stroke="#525252" strokeWidth={1} markerEnd="url(#gpt-arrow)" />

        <ArchBlock
          cx={cx}
          y={lnY}
          w={boxW}
          h={boxH}
          color="#f472b6"
          label="final layer norm"
          sublabel={`γ, β ∈ R^${D_MODEL}`}
          active={selected === 'ln_f'}
          onClick={() => onSelect('ln_f')}
        />
        <line x1={cx} y1={lnY + boxH} x2={cx} y2={lmY} stroke="#525252" strokeWidth={1} markerEnd="url(#gpt-arrow)" />

        <ArchBlock
          cx={cx}
          y={lmY}
          w={boxW}
          h={boxH}
          color="#4ade80"
          label="lm_head (tied W_E)"
          sublabel={`(${D_MODEL}, ${VOCAB})`}
          active={selected === 'lm_head'}
          onClick={() => onSelect('lm_head')}
        />

        <text x={cx} y={H - 6} textAnchor="middle" fontSize="10" fill="#a1a1aa" fontFamily="JetBrains Mono, monospace">
          logits → softmax → next token
        </text>
      </svg>
    </div>
  )
}

function ArchBlock({
  cx,
  y,
  w,
  h,
  color,
  label,
  sublabel,
  active,
  onClick,
}: {
  cx: number
  y: number
  w: number
  h: number
  color: string
  label: string
  sublabel: string
  active: boolean
  onClick: () => void
}) {
  return (
    <g onClick={onClick} style={{ cursor: 'pointer' }}>
      <rect
        x={cx - w / 2}
        y={y}
        width={w}
        height={h}
        rx={6}
        fill={active ? color : '#141420'}
        fillOpacity={active ? 0.18 : 1}
        stroke={color}
        strokeWidth={active ? 2 : 1.1}
      />
      <text x={cx} y={y + h / 2 - 2} textAnchor="middle" fontSize="11" fill={active ? color : '#e5e7eb'} fontFamily="JetBrains Mono, monospace">
        {label}
      </text>
      <text x={cx} y={y + h / 2 + 11} textAnchor="middle" fontSize="9" fill="#71717a" fontFamily="JetBrains Mono, monospace">
        {sublabel}
      </text>
    </g>
  )
}
