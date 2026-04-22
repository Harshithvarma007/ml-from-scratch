'use client'

import { notFound } from 'next/navigation'
import WIDGET_REGISTRY from '@/components/lesson/content/widget-registry'
import WidgetShareBar from '@/components/ui/WidgetShareBar'

export default function WidgetClientPage({ name }: { name: string }) {
  const Widget = WIDGET_REGISTRY[name]

  if (!Widget) notFound()

  return (
    <div className="min-h-screen bg-dark-bg flex flex-col">
      <WidgetShareBar widgetName={name} />
      <main className="flex-1 flex items-center justify-center p-6">
        <div className="w-full max-w-4xl">
          <Widget />
        </div>
      </main>
    </div>
  )
}
