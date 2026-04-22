import type { MetadataRoute } from 'next'
import { flatLessons, roadmap } from '@/lib/roadmap'

// Change this to the deployed domain when shipping. Sitemap paths are relative
// to the host Next prepends from the current request URL, so a relative base is
// fine for build-time static generation.
const BASE = process.env.NEXT_PUBLIC_SITE_URL ?? 'https://ml.harshithvarma.in'

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date()

  const home: MetadataRoute.Sitemap[number] = {
    url: `${BASE}/`,
    lastModified: now,
    changeFrequency: 'weekly',
    priority: 1,
  }

  const sectionRoots: MetadataRoute.Sitemap = roadmap.map((s) => ({
    url: `${BASE}/learn/${s.slug}`,
    lastModified: now,
    changeFrequency: 'weekly',
    priority: 0.8,
  }))

  const lessons: MetadataRoute.Sitemap = flatLessons().map((l) => ({
    url: `${BASE}/learn/${l.sectionSlug}/${l.slug}`,
    lastModified: now,
    changeFrequency: 'monthly',
    priority: 0.7,
  }))

  return [home, ...sectionRoots, ...lessons]
}
