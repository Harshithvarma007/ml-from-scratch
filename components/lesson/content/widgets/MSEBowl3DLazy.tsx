'use client'

import { lazyViz } from './lazyViz'

export default lazyViz(() => import('./MSEBowl3D'), {
  label: 'MSE loss bowl — loading…',
})
