'use client'

import { lazyViz } from './lazyViz'

// Client-side dynamic wrapper that code-splits the three.js bundle. Imported
// from server-rendered lesson files, which can't use `next/dynamic({ ssr: false })`
// directly. The 3D canvas only ships when this widget is actually on screen.
export default lazyViz(() => import('./LossSurface3D'), {
  label: '3D loss surface — loading…',
})
