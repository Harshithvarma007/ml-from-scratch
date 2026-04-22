'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { PLAYBACK_MS } from '@/lib/constants/timing'

interface Options {
  /** Interval between steps in ms. Defaults to PLAYBACK_MS (600). */
  interval?: number
  /** When reaching totalSteps, loop back to 0. Default: stop at end. */
  loop?: boolean
  /** Skip autoplay if the user's OS has prefers-reduced-motion. Default: true. */
  respectReducedMotion?: boolean
}

interface Playback {
  /** Currently displayed step, 0..totalSteps-1. */
  step: number
  /** Whether autoplay is currently running. */
  playing: boolean
  /** Toggle play/pause. Auto-resets from end when starting if not looping. */
  togglePlay: () => void
  /** Stop playback. */
  pause: () => void
  /** Set the current step directly — used by scrubbers and step-forward buttons. */
  setStep: (n: number) => void
  /** Reset to step 0 and pause. */
  reset: () => void
}

// Shared playback state machine for stepped widgets (BPTT, DDPM, REINFORCE,
// specs, continuous batching, etc.). Replaces ad-hoc setInterval loops that
// all drifted out of sync on timing, cleanup, and reduced-motion.
export default function usePlayback(totalSteps: number, opts: Options = {}): Playback {
  const { interval = PLAYBACK_MS, loop = false, respectReducedMotion = true } = opts
  const [step, setStepState] = useState(0)
  const [playing, setPlaying] = useState(false)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Clamp step when totalSteps changes (e.g. a slider changed the problem size).
  useEffect(() => {
    setStepState((s) => (s >= totalSteps ? Math.max(0, totalSteps - 1) : s))
  }, [totalSteps])

  const clearTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  useEffect(() => {
    if (!playing) {
      clearTimer()
      return
    }
    timerRef.current = setInterval(() => {
      setStepState((s) => {
        const next = s + 1
        if (next >= totalSteps) {
          if (loop) return 0
          // Stop at end
          setPlaying(false)
          return totalSteps - 1
        }
        return next
      })
    }, interval)
    return clearTimer
  }, [playing, totalSteps, interval, loop])

  const togglePlay = useCallback(() => {
    if (
      respectReducedMotion &&
      typeof window !== 'undefined' &&
      window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    ) {
      // Don't auto-advance for users who asked us not to — just step forward once.
      setStepState((s) => (s + 1 >= totalSteps ? (loop ? 0 : s) : s + 1))
      return
    }
    setPlaying((p) => {
      if (!p && step >= totalSteps - 1 && !loop) {
        // Restart from the beginning when pressing play at the end.
        setStepState(0)
      }
      return !p
    })
  }, [step, totalSteps, loop, respectReducedMotion])

  const pause = useCallback(() => setPlaying(false), [])
  const reset = useCallback(() => {
    setPlaying(false)
    setStepState(0)
  }, [])
  const setStep = useCallback(
    (n: number) => setStepState(Math.max(0, Math.min(totalSteps - 1, n))),
    [totalSteps],
  )

  return { step, playing, togglePlay, pause, setStep, reset }
}
