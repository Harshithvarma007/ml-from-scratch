'use client'

import dynamic from 'next/dynamic'
import type { ComponentType } from 'react'
import WidgetFrame from './WidgetFrame'

// Factory for code-split 3D / heavy client-only widgets.
// Wraps next/dynamic with a WidgetFrame-backed loading state so every
// lazy widget gets the same "initializing WebGL scene…" skeleton without
// each one hand-rolling it.
//
// Usage (from a lesson file, not from the widget's own module):
//
//   const LossSurface3D = lazyViz(
//     () => import('./LossSurface3D'),
//     { label: '3D loss surface — loading…' }
//   )
export function lazyViz<P extends object = {}>(
  loader: () => Promise<{ default: ComponentType<P> }>,
  { label, aspect = 'wide' }: { label: string; aspect?: 'wide' | 'square' | 'tall' } = {
    label: 'loading…',
  },
): ComponentType<P> {
  return dynamic(loader, {
    ssr: false,
    loading: () => (
      <WidgetFrame label={label} aspect={aspect}>
        <div className="absolute inset-0 flex items-center justify-center font-mono text-[11px] text-dark-text-disabled">
          initializing WebGL scene…
        </div>
      </WidgetFrame>
    ),
  })
}
