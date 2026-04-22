'use client'

import { useState } from 'react'
import WidgetFrame, { Slider, Readout } from './WidgetFrame'
import { cn } from '@/lib/utils'

// Show an image (synthetic stylized "photo") being chopped into patches.
// Slider controls patch size. Token count + d_model dims update. Highlight
// one patch to show it being flattened into a sequence vector.

const IMG_SIZE = 96

function makePhoto(): number[][] {
  // Simple gradient + circle to look "image-ish"
  const img: number[][] = []
  for (let y = 0; y < IMG_SIZE; y++) {
    const row: number[] = []
    for (let x = 0; x < IMG_SIZE; x++) {
      const cx = x - IMG_SIZE / 2
      const cy = y - IMG_SIZE / 2
      const sky = y < IMG_SIZE * 0.5 ? 0.3 + 0.4 * (1 - y / (IMG_SIZE * 0.5)) : 0.1
      const ground = y >= IMG_SIZE * 0.5 ? 0.2 + 0.3 * ((y - IMG_SIZE * 0.5) / (IMG_SIZE * 0.5)) : 0
      const sun =
        cx * cx + (cy + 20) * (cy + 20) < 80 ? 1 : 0
      const hill =
        Math.abs(cx + 10) < 15 && y > IMG_SIZE * 0.4 && y < IMG_SIZE * 0.7
          ? 0.5
          : 0
      row.push(Math.max(sky + ground, sun, hill))
    }
    img.push(row)
  }
  return img
}

const IMG = makePhoto()

export default function ImageToPatches() {
  const [patchSize, setPatchSize] = useState(16)
  const [highlight, setHighlight] = useState<{ r: number; c: number }>({ r: 2, c: 3 })

  const nRows = Math.floor(IMG_SIZE / patchSize)
  const nCols = Math.floor(IMG_SIZE / patchSize)
  const numPatches = nRows * nCols
  const patchDim = patchSize * patchSize * 3 // 3 channels
  const dModel = 768
  const seqLen = numPatches + 1 // + CLS

  // Extract highlighted patch pixel-by-pixel into a flat vector preview
  const flatPatch: number[] = []
  for (let y = 0; y < patchSize; y++) {
    for (let x = 0; x < patchSize; x++) {
      const iy = highlight.r * patchSize + y
      const ix = highlight.c * patchSize + x
      if (iy < IMG_SIZE && ix < IMG_SIZE) flatPatch.push(IMG[iy][ix])
      else flatPatch.push(0)
    }
  }

  return (
    <WidgetFrame
      widgetName="ImageToPatches"
      label="image → patches → tokens"
      right={<span className="font-mono">image ({IMG_SIZE}×{IMG_SIZE}) · patch size P</span>}
      aspect="wide"
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="patch size P"
            value={patchSize}
            min={8}
            max={32}
            step={8}
            onChange={(v) => setPatchSize(Math.round(v / 8) * 8)}
            format={(v) => `${Math.round(v / 8) * 8}×${Math.round(v / 8) * 8}`}
            accent="accent-term-amber"
          />
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="#patches" value={String(numPatches)} />
            <Readout label="seq len" value={`${seqLen} (+CLS)`} accent="text-term-amber" />
            <Readout label="patch dim" value={String(patchDim)} />
            <Readout label="d_model" value={String(dModel)} />
          </div>
        </div>
      }
    >
      <div className="absolute inset-0 p-5 grid grid-cols-[auto_1fr] gap-6 items-center overflow-auto">
        {/* Left: image with grid overlay */}
        <div className="flex flex-col items-center gap-2">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            input image · click a patch
          </div>
          <div
            className="relative border border-dark-border rounded overflow-hidden"
            style={{ width: `${IMG_SIZE * 2.5}px`, height: `${IMG_SIZE * 2.5}px` }}
          >
            <div
              className="grid absolute inset-0"
              style={{ gridTemplateColumns: `repeat(${IMG_SIZE}, 1fr)`, gap: 0 }}
            >
              {IMG.flatMap((row, y) =>
                row.map((v, x) => (
                  <div
                    key={`${y}-${x}`}
                    style={{ backgroundColor: `rgba(${Math.floor(v * 220)}, ${Math.floor(v * 180 + 40)}, ${Math.floor(v * 140 + 30)}, 1)` }}
                  />
                )),
              )}
            </div>
            {/* Patch grid overlay */}
            <div
              className="absolute inset-0 grid"
              style={{
                gridTemplateColumns: `repeat(${nCols}, 1fr)`,
                gridTemplateRows: `repeat(${nRows}, 1fr)`,
                gap: 0,
              }}
            >
              {Array.from({ length: nRows }).flatMap((_, r) =>
                Array.from({ length: nCols }).map((_, c) => (
                  <button
                    key={`${r}-${c}`}
                    onClick={() => setHighlight({ r, c })}
                    className={cn(
                      'border border-dark-border/40 hover:bg-white/10 transition-colors',
                      highlight.r === r && highlight.c === c && 'bg-term-amber/20 border-term-amber',
                    )}
                  />
                )),
              )}
            </div>
          </div>
        </div>

        {/* Right: flattened patch preview */}
        <div className="flex flex-col gap-3">
          <div className="text-[10px] font-mono uppercase tracking-wider text-dark-text-disabled">
            selected patch (row {highlight.r}, col {highlight.c}) → flattened token
          </div>
          <div
            className="inline-grid border border-term-amber/50 rounded overflow-hidden"
            style={{ gridTemplateColumns: `repeat(${patchSize}, 12px)`, gap: 0, width: `${patchSize * 12}px` }}
          >
            {flatPatch.map((v, i) => (
              <div
                key={i}
                style={{
                  width: 12,
                  height: 12,
                  backgroundColor: `rgba(${Math.floor(v * 220)}, ${Math.floor(v * 180 + 40)}, ${Math.floor(v * 140 + 30)}, 1)`,
                }}
              />
            ))}
          </div>
          <div className="text-[11px] font-mono text-dark-text-muted">
            raw patch: {patchSize}×{patchSize}×3 = {patchDim} numbers
          </div>
          <div className="text-dark-accent">↓ nn.Linear({patchDim}, {dModel})</div>
          <div
            className="inline-flex rounded overflow-hidden"
            style={{ gap: 1 }}
          >
            {Array.from({ length: 24 }).map((_, i) => (
              <div
                key={i}
                className="bg-term-amber/60"
                style={{ width: 14, height: 28, opacity: 0.3 + Math.random() * 0.6 }}
              />
            ))}
            <span className="text-dark-text-disabled ml-2 self-center text-[10px]">… ×{dModel}</span>
          </div>
          <div className="text-[11px] font-mono text-dark-text-muted mt-2">
            each patch → one {dModel}-dim token · transformer sees a sequence of {seqLen} tokens (incl. CLS)
          </div>
        </div>
      </div>
    </WidgetFrame>
  )
}
