import type { Metadata } from 'next'
import { WIDGET_NAMES } from '@/components/lesson/content/widget-registry'
import WidgetClientPage from './WidgetClientPage'

export const dynamicParams = true

export function generateStaticParams() {
  return WIDGET_NAMES.map((name) => ({ name }))
}

export async function generateMetadata({
  params,
}: {
  params: { name: string }
}): Promise<Metadata> {
  return {
    title: `${params.name} · ML from Scratch`,
    description: `Interactive ${params.name} widget — ML from Scratch`,
  }
}

export default function WidgetPage({ params }: { params: { name: string } }) {
  return <WidgetClientPage name={params.name} />
}
