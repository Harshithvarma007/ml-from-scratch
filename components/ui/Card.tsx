import { cn } from '@/lib/utils'
import type { HTMLAttributes, ReactNode } from 'react'

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode
  elevated?: boolean
}

// .term-panel bordered inset block — the universal surface in this app.
export default function Card({ children, className, elevated, ...rest }: CardProps) {
  return (
    <div
      className={cn(elevated ? 'term-panel-elevated' : 'term-panel', className)}
      {...rest}
    >
      {children}
    </div>
  )
}

interface CardHeaderProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode
}

export function CardHeader({ children, className, ...rest }: CardHeaderProps) {
  return (
    <div className={cn('term-panel-header', className)} {...rest}>
      {children}
    </div>
  )
}
