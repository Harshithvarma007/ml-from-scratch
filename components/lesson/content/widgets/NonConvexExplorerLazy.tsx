'use client'

import { lazyViz } from './lazyViz'

export default lazyViz(() => import('./NonConvexExplorer'), {
  label: 'non-convex surface — loading…',
})
