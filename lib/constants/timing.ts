// Shared animation / playback timing constants. Widgets should prefer these
// over literal millisecond values so cadence stays consistent across the site
// and a single number change reflows every widget. Note: these are *defaults* —
// a widget can still override via a constructor arg (e.g. `usePlayback(n, 400)`).

/** Default step interval for auto-advancing playback widgets (BPTT, DDPM, etc.). */
export const PLAYBACK_MS = 600

/** Per-step transition duration for scrub-style widgets. */
export const STEP_MS = 250

/** Fade in/out for overlays, popovers, reveal-on-scroll sections. */
export const FADE_MS = 200

/** Tooltip/popover show delay before appearing on hover. */
export const HOVER_DELAY_MS = 120
