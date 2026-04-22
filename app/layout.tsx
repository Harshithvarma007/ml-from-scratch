import type { Metadata } from 'next'
import CommandPalette from '@/components/search/CommandPalette'
import CommandPaletteHint from '@/components/search/CommandPaletteHint'
import './globals.css'

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.harshithvarma.in'

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: 'ML from Scratch — Derive it yourself',
    template: '%s · ML from Scratch',
  },
  description:
    'From gradient descent to GPT — every equation proven, every line of code built from zero. 75 lessons, 168+ interactive widgets, no hand-waving.',
  keywords: [
    'machine learning',
    'deep learning',
    'neural networks',
    'transformers',
    'GPT',
    'from scratch',
    'tutorial',
    'PyTorch',
    'NumPy',
    'backpropagation',
    'interactive',
  ],
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: SITE_URL,
    siteName: 'ML from Scratch',
    title: 'ML from Scratch — Derive it yourself',
    description:
      'From gradient descent to GPT — every equation proven, every line of code built from zero. 75 lessons, 168+ interactive widgets.',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'ML from Scratch — Derive it yourself',
    description:
      'From gradient descent to GPT — every equation proven, every line of code built from zero. 75 lessons, 168+ interactive widgets.',
  },
  icons: {
    icon: [
      {
        url:
          'data:image/svg+xml,' +
          encodeURIComponent(
            `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="6" fill="#0a0a0a"/><text x="16" y="22" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="18" font-weight="600" fill="#a78bfa">ml</text></svg>`
          ),
      },
    ],
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-dark-bg text-dark-text-primary min-h-screen">
        {children}
        <CommandPalette />
        <CommandPaletteHint />
      </body>
    </html>
  )
}
