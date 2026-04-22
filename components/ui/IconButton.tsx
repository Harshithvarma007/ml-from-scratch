import { cn } from '@/lib/utils'
import { forwardRef, type ButtonHTMLAttributes } from 'react'

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  tone?: 'ghost' | 'accent'
}

const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ tone = 'ghost', className, children, ...rest }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center p-1 rounded transition-colors',
          tone === 'ghost' &&
            'text-dark-text-disabled hover:text-dark-text-secondary hover:bg-white/[0.05]',
          tone === 'accent' && 'text-dark-accent hover:bg-dark-accent/10',
          className
        )}
        {...rest}
      >
        {children}
      </button>
    )
  }
)
IconButton.displayName = 'IconButton'
export default IconButton
