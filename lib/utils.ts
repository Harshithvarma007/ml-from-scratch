import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function lessonHref(sectionSlug: string, lessonSlug: string): string {
  return `/learn/${sectionSlug}/${lessonSlug}`
}

export function sectionHref(sectionSlug: string): string {
  return `/learn/${sectionSlug}`
}
