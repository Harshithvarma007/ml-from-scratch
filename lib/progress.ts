// Ultra-lightweight localStorage-backed progress tracker.
// Keyed by `${sectionSlug}/${lessonSlug}`.

const STORAGE_KEY = 'mlfs:progress:v1'

export type Status = 'not-started' | 'in-progress' | 'completed'

type ProgressMap = Record<string, Status>

function key(sectionSlug: string, lessonSlug: string) {
  return `${sectionSlug}/${lessonSlug}`
}

function read(): ProgressMap {
  if (typeof window === 'undefined') return {}
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    return raw ? (JSON.parse(raw) as ProgressMap) : {}
  } catch {
    return {}
  }
}

function write(map: ProgressMap) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(map))
    window.dispatchEvent(new Event('mlfs:progress-updated'))
  } catch {
    // Swallow — quota exceeded or private browsing
  }
}

export function getStatus(sectionSlug: string, lessonSlug: string): Status {
  return read()[key(sectionSlug, lessonSlug)] ?? 'not-started'
}

export function setStatus(sectionSlug: string, lessonSlug: string, status: Status) {
  const map = read()
  if (status === 'not-started') {
    delete map[key(sectionSlug, lessonSlug)]
  } else {
    map[key(sectionSlug, lessonSlug)] = status
  }
  write(map)
}

export function cycleStatus(sectionSlug: string, lessonSlug: string): Status {
  const current = getStatus(sectionSlug, lessonSlug)
  const next: Status =
    current === 'not-started' ? 'in-progress' : current === 'in-progress' ? 'completed' : 'not-started'
  setStatus(sectionSlug, lessonSlug, next)
  return next
}

export function allProgress(): ProgressMap {
  return read()
}
