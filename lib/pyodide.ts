'use client'

// Pyodide singleton loader.
//
// We load Pyodide from the jsdelivr CDN so our own Vercel/Netlify bandwidth
// only covers the site itself, not the ~10 MB Python runtime + ~4 MB NumPy
// package. That same CDN path is used by hundreds of other Pyodide sites,
// which means browser + edge caches are already warm for many visitors.
//
// The loader is lazy: nothing loads until the first RunnableCodeBlock on the
// page actually presses "run". The boot promise is memoized so if two blocks
// run at once, they share the same boot instead of racing.

const PYODIDE_VERSION = '0.26.4'
const CDN_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`

// The subset of the Pyodide surface we actually touch. Keeping this narrow
// lets us upgrade Pyodide versions with confidence — if they rename
// `setStdout` we'll see a compile error at the shim, not at every caller.
export interface PyodideRuntime {
  runPythonAsync: (code: string) => Promise<unknown>
  setStdout: (opts: { batched: (s: string) => void }) => void
  setStderr: (opts: { batched: (s: string) => void }) => void
  loadPackage: (
    names: string | string[],
    opts?: { messageCallback?: (m: string) => void },
  ) => Promise<void>
  globals: {
    clear: () => void
  }
}

declare global {
  interface Window {
    loadPyodide?: (opts?: { indexURL?: string }) => Promise<PyodideRuntime>
  }
}

let bootPromise: Promise<PyodideRuntime> | null = null
let scriptPromise: Promise<void> | null = null

function loadPyodideScript(): Promise<void> {
  if (!scriptPromise) {
    scriptPromise = new Promise<void>((resolve, reject) => {
      if (typeof window === 'undefined') {
        reject(new Error('Pyodide requires a browser environment.'))
        return
      }
      // If another instance of this loader already injected the script, just
      // wait for it to finish booting.
      if (window.loadPyodide) {
        resolve()
        return
      }
      const src = `${CDN_URL}pyodide.js`
      const existing = document.querySelector<HTMLScriptElement>(
        `script[data-pyodide-loader="1"]`,
      )
      if (existing) {
        existing.addEventListener('load', () => resolve())
        existing.addEventListener('error', () => reject(new Error('Pyodide script failed to load')))
        return
      }
      const s = document.createElement('script')
      s.src = src
      s.async = true
      s.dataset.pyodideLoader = '1'
      s.onload = () => resolve()
      s.onerror = () => reject(new Error('Pyodide script failed to load'))
      document.head.appendChild(s)
    })
  }
  return scriptPromise
}

// Memoized boot. First call kicks off the download + WASM instantiation;
// subsequent calls resolve to the same runtime instance.
export function getPyodide(): Promise<PyodideRuntime> {
  if (!bootPromise) {
    bootPromise = (async () => {
      await loadPyodideScript()
      if (!window.loadPyodide) {
        throw new Error('Pyodide loaded but window.loadPyodide is missing.')
      }
      return window.loadPyodide({ indexURL: CDN_URL })
    })()
  }
  return bootPromise
}

// Packages already loaded in this session. Pyodide's loadPackage is
// idempotent but is not cheap to call repeatedly, so we short-circuit here.
const loadedPackages = new Set<string>()

// Scan source for imports we know how to satisfy from Pyodide's pre-built
// package wheels. We deliberately handle a tiny allow-list (numpy only for
// v1) — silently trying to load scipy/sklearn/matplotlib would surprise a
// reader with a 30 MB download they didn't ask for.
const KNOWN_PACKAGES: Array<{ name: string; pattern: RegExp }> = [
  { name: 'numpy', pattern: /^\s*(?:import numpy|from numpy)/m },
]

export interface PackageLoadResult {
  /** Names of packages this run kicked off loading for. */
  requested: string[]
}

export async function ensurePackagesForSource(
  source: string,
  runtime: PyodideRuntime,
  onStatus?: (msg: string) => void,
): Promise<PackageLoadResult> {
  const needed: string[] = []
  for (const { name, pattern } of KNOWN_PACKAGES) {
    if (!loadedPackages.has(name) && pattern.test(source)) {
      needed.push(name)
    }
  }
  if (needed.length > 0) {
    onStatus?.(`loading ${needed.join(', ')}…`)
    await runtime.loadPackage(needed)
    for (const n of needed) loadedPackages.add(n)
  }
  return { requested: needed }
}

// Reset module-level globals between runs. Two back-to-back runs of the
// same block shouldn't leak variable bindings from the first — that would
// surprise readers ("why is x still 5 after I changed the code to x=3?").
export function resetPythonGlobals(runtime: PyodideRuntime): void {
  runtime.globals.clear()
}
