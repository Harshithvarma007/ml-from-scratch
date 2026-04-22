import {
  Sigma,
  Network,
  Flame,
  Gauge,
  Type,
  Eye,
  Sparkles,
  Image,
  Repeat,
  SlidersHorizontal,
  Boxes,
  Waves,
  Target,
  Zap,
  type LucideIcon,
} from 'lucide-react'
import type { SectionIcon as IconName } from '@/lib/roadmap'
import { cn } from '@/lib/utils'

const ICON_MAP: Record<IconName, LucideIcon> = {
  sigma: Sigma,
  network: Network,
  flame: Flame,
  gauge: Gauge,
  type: Type,
  eye: Eye,
  sparkles: Sparkles,
  image: Image,
  repeat: Repeat,
  sliders: SlidersHorizontal,
  boxes: Boxes,
  waves: Waves,
  target: Target,
  zap: Zap,
}

interface SectionIconProps {
  name: IconName
  className?: string
}

// Centralized mapping so section icons stay consistent across home, sidebar, and lesson header.
export default function SectionIcon({ name, className }: SectionIconProps) {
  const Icon = ICON_MAP[name]
  return <Icon className={cn('w-3.5 h-3.5', className)} strokeWidth={1.75} />
}
