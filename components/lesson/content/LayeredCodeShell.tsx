'use client'

import { Children, useState, type ReactNode } from 'react'
import { Layers } from 'lucide-react'
import { cn } from '@/lib/utils'

interface LayeredCodeShellProps {
  labels: string[]
  defaultIndex?: number
  children: ReactNode
}

// Client shell that owns the active-tab state for a <LayeredCode/>. All
// layers are mounted at once; inactive ones are marked `hidden` so that a
// runnable layer's stdout + edit state survives tab switches. React Server
// Components are allowed as children because we just toggle their wrappers.
export default function LayeredCodeShell({
  labels,
  defaultIndex = 0,
  children,
}: LayeredCodeShellProps) {
  const [active, setActive] = useState(defaultIndex)
  const layers = Children.toArray(children)

  return (
    <figure className="my-5">
      <div
        role="tablist"
        aria-label="Code layer"
        className="flex items-center gap-0.5 border-b border-dark-border"
      >
        <Layers className="w-3 h-3 text-dark-accent mx-2 shrink-0" strokeWidth={2} />
        {labels.map((label, i) => (
          <button
            key={i}
            type="button"
            role="tab"
            aria-selected={i === active}
            aria-controls={`layered-panel-${i}`}
            id={`layered-tab-${i}`}
            onClick={() => setActive(i)}
            className={cn(
              'px-3 py-1.5 text-[10px] font-mono uppercase tracking-wider',
              'border-b-2 -mb-px transition-colors',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-dark-accent focus-visible:ring-inset',
              i === active
                ? 'text-dark-accent border-dark-accent'
                : 'text-dark-text-disabled border-transparent hover:text-dark-text-secondary hover:border-dark-border-hover',
            )}
          >
            {label}
          </button>
        ))}
      </div>

      {layers.map((child, i) => (
        <div
          key={i}
          role="tabpanel"
          id={`layered-panel-${i}`}
          aria-labelledby={`layered-tab-${i}`}
          hidden={i !== active}
        >
          {child}
        </div>
      ))}
    </figure>
  )
}
