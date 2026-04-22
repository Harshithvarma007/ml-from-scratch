'use client'

import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber'
import { OrbitControls, Line } from '@react-three/drei'
import { Suspense, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import WidgetFrame, { Slider, Button, Readout } from './WidgetFrame'
import { Play, RotateCcw, MousePointer2 } from 'lucide-react'

// The MSE loss for a 2-parameter linear regression is a quadratic bowl in
// (w, b) space. Show it as a 3D surface. Place a marble, release, watch it
// roll to the minimum. Mark the closed-form optimum with a purple pin — the
// marble always lands there given enough steps. That equivalence is the
// punchline of the lesson.

// Tiny fixed dataset: y = 1.5x + 0.3 with noise.
function makeData(): Array<[number, number]> {
  return [
    [-1.8, -2.1],
    [-1.2, -1.3],
    [-0.5, -0.4],
    [0.1, 0.6],
    [0.9, 1.7],
    [1.4, 2.2],
    [2.1, 3.5],
    [2.6, 4.1],
  ]
}

function mse(w: number, b: number, data: Array<[number, number]>): number {
  let s = 0
  for (const [x, y] of data) {
    const r = y - (w * x + b)
    s += r * r
  }
  return s / data.length
}

// Closed-form OLS for y = wx + b:  w = cov(x,y)/var(x),  b = ȳ - w·x̄
function closedForm(data: Array<[number, number]>): [number, number] {
  const n = data.length
  const xbar = data.reduce((s, [x]) => s + x, 0) / n
  const ybar = data.reduce((s, [, y]) => s + y, 0) / n
  const cov = data.reduce((s, [x, y]) => s + (x - xbar) * (y - ybar), 0)
  const varX = data.reduce((s, [x]) => s + (x - xbar) ** 2, 0)
  const w = cov / varX
  const b = ybar - w * xbar
  return [w, b]
}

const W_MIN = -1
const W_MAX = 3
const B_MIN = -2
const B_MAX = 2
const SURFACE_RES = 48

function Surface({
  data,
  onPick,
}: {
  data: Array<[number, number]>
  onPick: (w: number, b: number) => void
}) {
  const geometry = useMemo(() => {
    const geo = new THREE.PlaneGeometry(
      W_MAX - W_MIN,
      B_MAX - B_MIN,
      SURFACE_RES,
      SURFACE_RES,
    )
    const pos = geo.attributes.position
    for (let i = 0; i < pos.count; i++) {
      const wLocal = pos.getX(i) + (W_MAX + W_MIN) / 2
      const bLocal = pos.getY(i) + (B_MAX + B_MIN) / 2
      const z = Math.log1p(mse(wLocal, bLocal, data)) // compress dynamic range
      pos.setZ(i, z)
    }
    geo.computeVertexNormals()
    return geo
  }, [data])

  return (
    <group
      rotation={[-Math.PI / 2, 0, 0]}
      position={[-(W_MAX + W_MIN) / 2, 0, -(B_MAX + B_MIN) / 2]}
    >
      <mesh
        geometry={geometry}
        position={[(W_MAX + W_MIN) / 2, (B_MAX + B_MIN) / 2, 0]}
        onPointerDown={(e: ThreeEvent<PointerEvent>) => {
          e.stopPropagation()
          const p = e.point
          onPick(p.x, p.z)
        }}
      >
        <meshStandardMaterial
          color="#1a1830"
          opacity={0.82}
          transparent
          metalness={0.12}
          roughness={0.78}
          side={THREE.DoubleSide}
        />
      </mesh>
      <mesh geometry={geometry} position={[(W_MAX + W_MIN) / 2, (B_MAX + B_MIN) / 2, 0]}>
        <meshBasicMaterial color="#a78bfa" wireframe transparent opacity={0.18} />
      </mesh>
    </group>
  )
}

function Marble({ position, color }: { position: [number, number, number]; color: string }) {
  return (
    <mesh position={position}>
      <sphereGeometry args={[0.08, 24, 24]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.45} />
    </mesh>
  )
}

function OptimumPin({ w, b, loss }: { w: number; b: number; loss: number }) {
  const y = Math.log1p(loss) + 0.1
  return (
    <>
      <mesh position={[w, y, b]}>
        <sphereGeometry args={[0.07, 16, 16]} />
        <meshStandardMaterial color="#a78bfa" emissive="#a78bfa" emissiveIntensity={0.8} />
      </mesh>
      <Line
        points={[
          [w, 0, b],
          [w, y, b],
        ]}
        color="#a78bfa"
        lineWidth={1}
        transparent
        opacity={0.5}
      />
    </>
  )
}

interface State {
  w: number
  b: number
  running: boolean
  trail: THREE.Vector3[]
}

function Sim({
  lr,
  data,
  state,
  setState,
}: {
  lr: number
  data: Array<[number, number]>
  state: State
  setState: React.Dispatch<React.SetStateAction<State>>
}) {
  const tickRef = useRef(0)
  useFrame((_, delta) => {
    if (!state.running) return
    tickRef.current += delta
    if (tickRef.current < 0.05) return
    tickRef.current = 0

    setState((s) => {
      if (!s.running) return s
      // Analytical gradient of MSE for y = w*x + b over this data:
      //   dMSE/dw = -2/N * Σ x(y - (wx+b))
      //   dMSE/db = -2/N * Σ (y - (wx+b))
      let gw = 0
      let gb = 0
      const N = data.length
      for (const [x, y] of data) {
        const r = y - (s.w * x + s.b)
        gw += -2 * x * r
        gb += -2 * r
      }
      gw /= N
      gb /= N
      const nw = s.w - lr * gw
      const nb = s.b - lr * gb
      const nloss = mse(nw, nb, data)
      const magnitude = Math.hypot(gw, gb)
      const converged = magnitude < 0.005
      const next = new THREE.Vector3(nw, Math.log1p(nloss) + 0.05, nb)
      return {
        w: nw,
        b: nb,
        running: !converged,
        trail: [...s.trail, next].slice(-400),
      }
    })
  })
  return null
}

function HoverMarker({ onHover }: { onHover: (p: THREE.Vector3 | null) => void }) {
  const { camera, mouse, raycaster, scene } = useThree()
  useFrame(() => {
    raycaster.setFromCamera(mouse, camera)
    const hits = raycaster.intersectObjects(scene.children, true)
    if (hits.length > 0) {
      const p = hits[0].point
      if (p.x >= W_MIN && p.x <= W_MAX && p.z >= B_MIN && p.z <= B_MAX) {
        onHover(p)
        return
      }
    }
    onHover(null)
  })
  return null
}

export default function MSEBowl3D() {
  const data = useMemo(() => makeData(), [])
  const [wStar, bStar] = useMemo(() => closedForm(data), [data])
  const optimumLoss = useMemo(() => mse(wStar, bStar, data), [wStar, bStar, data])

  const [lr, setLr] = useState(0.18)
  const [state, setState] = useState<State>(() => {
    const w0 = -0.5
    const b0 = 1.5
    return {
      w: w0,
      b: b0,
      running: false,
      trail: [new THREE.Vector3(w0, Math.log1p(mse(w0, b0, data)) + 0.05, b0)],
    }
  })
  const [hover, setHover] = useState<THREE.Vector3 | null>(null)

  const placeMarble = (w: number, b: number) => {
    const cw = Math.max(W_MIN + 0.05, Math.min(W_MAX - 0.05, w))
    const cb = Math.max(B_MIN + 0.05, Math.min(B_MAX - 0.05, b))
    setState({
      w: cw,
      b: cb,
      running: false,
      trail: [new THREE.Vector3(cw, Math.log1p(mse(cw, cb, data)) + 0.05, cb)],
    })
  }

  const release = () => setState((s) => ({ ...s, running: true }))
  const reset = () => placeMarble(-0.5, 1.5)

  const curLoss = mse(state.w, state.b, data)
  const marbleY = Math.log1p(curLoss) + 0.08

  return (
    <WidgetFrame
      widgetName="MSEBowl3D"
      label="MSE loss surface — every valid (w, b) scored"
      right={
        <>
          <span className="font-mono">L(w, b) = (1/N) Σ (yᵢ − wxᵢ − b)²</span>
          <span className="text-dark-text-disabled">·</span>
          <span>click to place</span>
        </>
      }
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="α"
            value={lr}
            min={0.01}
            max={0.5}
            step={0.01}
            onChange={setLr}
            accent="accent-term-purple"
          />
          <div className="flex items-center gap-2">
            <Button onClick={release} variant="primary" disabled={state.running}>
              <Play className="w-3 h-3 inline -mt-px mr-1" /> release
            </Button>
            <Button onClick={reset}>
              <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
            </Button>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="w" value={state.w.toFixed(2)} />
            <Readout label="b" value={state.b.toFixed(2)} />
            <Readout label="loss" value={curLoss.toFixed(3)} accent="text-term-amber" />
            <Readout
              label="optimum"
              value={`(${wStar.toFixed(2)}, ${bStar.toFixed(2)})`}
              accent="text-term-purple"
            />
          </div>
        </div>
      }
    >
      <Canvas camera={{ position: [3.5, 3, 3.5], fov: 42 }} gl={{ antialias: true, alpha: true }}>
        <color attach="background" args={['#0a0a0a']} />
        <Suspense fallback={null}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[5, 8, 3]} intensity={0.85} />
          <pointLight position={[-4, 5, -4]} intensity={0.3} color="#a78bfa" />

          <Surface data={data} onPick={placeMarble} />

          {state.trail.length > 1 && (
            <Line points={state.trail} color="#fbbf24" lineWidth={2} transparent opacity={0.85} />
          )}
          <Marble position={[state.w, marbleY, state.b]} color="#fbbf24" />
          <OptimumPin w={wStar} b={bStar} loss={optimumLoss} />

          {hover && !state.running && (
            <mesh position={[hover.x, hover.y + 0.03, hover.z]}>
              <ringGeometry args={[0.12, 0.18, 24]} />
              <meshBasicMaterial color="#a78bfa" side={THREE.DoubleSide} />
            </mesh>
          )}

          <Sim lr={lr} data={data} state={state} setState={setState} />
          <HoverMarker onHover={setHover} />
          <OrbitControls
            enablePan={false}
            minDistance={3}
            maxDistance={10}
            maxPolarAngle={Math.PI / 2 - 0.1}
          />
        </Suspense>
      </Canvas>

      {!state.running && state.trail.length === 1 && (
        <div className="absolute top-3 left-3 flex items-center gap-1.5 text-[11px] font-mono text-dark-text-muted pointer-events-none">
          <MousePointer2 className="w-3 h-3" />
          <span>click anywhere on the bowl · the purple pin is the exact optimum</span>
        </div>
      )}
    </WidgetFrame>
  )
}
