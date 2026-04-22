'use client'

// 8 data shards (one per rectangle) read round-robin by a training loop.
// Animated pointer walks across shards; num_workers controls how many shards
// are "in flight" at once (shown as simultaneously-active). Shuffle seed
// permutes shard read order. Throughput is estimated from workers × shard
// size / per-batch time.

import { useEffect, useMemo, useRef, useState } from 'react'
import WidgetFrame, { Button, Readout } from './WidgetFrame'
import { Play, Pause, Shuffle } from 'lucide-react'
import { cn } from '@/lib/utils'

const N_SHARDS = 8
const SHARD_SIZE_MB = 128
const BATCHES_PER_SHARD = 40 // arbitrary; determines pointer granularity

function permute(n: number, seed: number): number[] {
  // deterministic Fisher-Yates with mulberry32
  const arr = Array.from({ length: n }, (_, i) => i)
  let s = seed
  const rng = () => {
    s += 0x6d2b79f5
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[arr[i], arr[j]] = [arr[j], arr[i]]
  }
  return arr
}

export default function ShardedDataLoader() {
  const [workers, setWorkers] = useState(2)
  const [playing, setPlaying] = useState(true)
  const [seed, setSeed] = useState(1)
  const [step, setStep] = useState(0) // in batches

  const order = useMemo(() => permute(N_SHARDS, seed), [seed])

  const maxStep = N_SHARDS * BATCHES_PER_SHARD

  // Animation loop
  const rafRef = useRef<number | null>(null)
  const lastTsRef = useRef<number | null>(null)
  useEffect(() => {
    if (!playing) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
      lastTsRef.current = null
      return
    }
    const tick = (ts: number) => {
      if (lastTsRef.current == null) lastTsRef.current = ts
      const dt = (ts - lastTsRef.current) / 1000
      lastTsRef.current = ts
      // speed scales with workers: one "step" per (0.18s / workers)
      const speed = 1 / (0.18 / workers)
      setStep((s) => (s + dt * speed) % maxStep)
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
      lastTsRef.current = null
    }
  }, [playing, workers, maxStep])

  const epochProgress = step / maxStep

  // Which shards are currently "in flight" (being read by workers)
  const activeShardPositions = new Set<number>()
  const curPos = Math.floor(step / BATCHES_PER_SHARD)
  for (let k = 0; k < workers; k++) {
    activeShardPositions.add((curPos + k) % N_SHARDS)
  }
  const activeShards = Array.from(activeShardPositions).map((p) => order[p])
  const activeShardSet = new Set(activeShards)

  // Per-shard "read" progress (how many batches have been consumed from each)
  const shardProgress = useMemo(() => {
    const p = new Array<number>(N_SHARDS).fill(0)
    const shardsRead = Math.floor(step / BATCHES_PER_SHARD)
    for (let i = 0; i < shardsRead; i++) p[order[i]] = 1
    if (curPos < N_SHARDS) {
      const partial = (step / BATCHES_PER_SHARD) - shardsRead
      p[order[curPos]] = partial
    }
    return p
  }, [step, order, curPos])

  const throughputMBps = (workers * SHARD_SIZE_MB) / 4.5 // fake but consistent: 4.5s / shard

  return (
    <WidgetFrame
      widgetName="ShardedDataLoader"
      label="sharded data loader — round-robin with num_workers"
      right={
        <span className="font-mono">
          {N_SHARDS} shards × {SHARD_SIZE_MB} MB · epoch {(epochProgress * 100).toFixed(0)}%
        </span>
      }
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <Button onClick={() => setPlaying(!playing)} variant="primary">
              <span className="inline-flex items-center gap-1">
                {playing ? <Pause size={11} /> : <Play size={11} />}
                {playing ? 'pause' : 'play'}
              </span>
            </Button>
            <Button onClick={() => setSeed((s) => s + 1)} variant="ghost">
              <span className="inline-flex items-center gap-1">
                <Shuffle size={11} /> shuffle
              </span>
            </Button>
          </div>
          <label className="flex items-center gap-3 flex-1 min-w-[220px] font-mono text-[12px]">
            <span className="text-dark-text-secondary whitespace-nowrap">num_workers</span>
            <input
              type="range"
              min={1}
              max={8}
              step={1}
              value={workers}
              onChange={(e) => setWorkers(Number(e.target.value))}
              className="flex-1 h-1 rounded-full bg-dark-border cursor-pointer accent-term-amber"
            />
            <span className="text-dark-text-primary tabular-nums w-8 text-right">{workers}</span>
          </label>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="throughput" value={`${throughputMBps.toFixed(0)} MB/s`} accent="text-term-green" />
            <Readout label="seed" value={String(seed)} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-4 flex flex-col gap-4 overflow-hidden">
        <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
          read order (seed {seed}): {order.map((o) => `#${o}`).join(' → ')}
        </div>

        {/* Shards visualization */}
        <div className="flex-1 flex flex-col gap-2 min-h-0">
          <div className="grid grid-cols-8 gap-2 flex-1">
            {Array.from({ length: N_SHARDS }).map((_, shardIdx) => {
              const orderPos = order.indexOf(shardIdx)
              const isActive = activeShardSet.has(shardIdx)
              const progress = shardProgress[shardIdx]
              const done = progress >= 1 && !isActive
              return (
                <div
                  key={shardIdx}
                  className={cn(
                    'relative rounded-md border overflow-hidden flex flex-col',
                    isActive
                      ? 'border-term-amber bg-term-amber/10'
                      : done
                        ? 'border-term-green/40 bg-term-green/5'
                        : 'border-dark-border bg-dark-surface-elevated/30',
                  )}
                >
                  {/* Header */}
                  <div className="flex items-center justify-between px-2 py-1 text-[10px] font-mono border-b border-dark-border/60">
                    <span className={cn('uppercase tracking-wider', isActive ? 'text-term-amber' : 'text-dark-text-disabled')}>
                      shard {shardIdx}
                    </span>
                    <span className="text-dark-text-disabled tabular-nums">#{orderPos + 1}</span>
                  </div>
                  {/* Batches grid (visual filling) */}
                  <div className="flex-1 p-1 grid grid-cols-5 gap-[1px] content-start">
                    {Array.from({ length: BATCHES_PER_SHARD }).map((_, bi) => {
                      const filled = progress * BATCHES_PER_SHARD > bi
                      return (
                        <div
                          key={bi}
                          className={cn(
                            'aspect-square rounded-[1px]',
                            filled
                              ? isActive
                                ? 'bg-term-amber/70'
                                : 'bg-term-green/50'
                              : 'bg-dark-bg',
                          )}
                        />
                      )
                    })}
                  </div>
                  {/* Progress bar */}
                  <div className="h-[3px] bg-dark-bg">
                    <div
                      className={cn(
                        'h-full transition-all',
                        isActive ? 'bg-term-amber' : done ? 'bg-term-green' : 'bg-dark-border',
                      )}
                      style={{ width: `${progress * 100}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Epoch strip */}
        <div className="flex flex-col gap-1">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            epoch pointer
          </div>
          <div className="relative h-4 bg-dark-surface-elevated/30 border border-dark-border rounded overflow-hidden">
            <div
              className="absolute top-0 bottom-0 bg-term-amber/40"
              style={{ width: `${epochProgress * 100}%` }}
            />
            <div
              className="absolute top-0 bottom-0 w-[2px] bg-term-amber"
              style={{ left: `${epochProgress * 100}%` }}
            />
            <div
              className="absolute inset-0 grid"
              style={{ gridTemplateColumns: `repeat(${N_SHARDS}, 1fr)` }}
            >
              {order.map((o, i) => (
                <div
                  key={i}
                  className="border-r border-dark-border/50 flex items-center justify-center text-[9px] font-mono text-dark-text-disabled"
                >
                  #{o}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
