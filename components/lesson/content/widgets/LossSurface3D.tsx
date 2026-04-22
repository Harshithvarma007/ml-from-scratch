'use client'

import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber'
import { OrbitControls, Line } from '@react-three/drei'
import { Suspense, useMemo, useRef, useState, useEffect } from 'react'
import * as THREE from 'three'
import WidgetFrame, { Slider, Button, Readout } from './WidgetFrame'
import { RotateCcw, Play, MousePointer2 } from 'lucide-react'

// f(x, z) = 0.2 * (x² + z²). We use x,z as the two parameters and y as height
// because Three.js uses Y as the up-axis by default.
const A = 0.2
function f(x: number, z: number) {
  return A * (x * x + z * z)
}
function grad(x: number, z: number): [number, number] {
  return [2 * A * x, 2 * A * z]
}

const EXTENT = 3
const MARBLE_R = 0.14

function Paraboloid({ onPick }: { onPick: (x: number, z: number) => void }) {
  const geometry = useMemo(() => {
    const seg = 60
    const geo = new THREE.PlaneGeometry(2 * EXTENT, 2 * EXTENT, seg, seg)
    const pos = geo.attributes.position
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i)
      const y = pos.getY(i) // plane-local Y before rotation
      pos.setZ(i, f(x, y))
    }
    geo.computeVertexNormals()
    return geo
  }, [])

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      {/* Solid translucent surface — acts as the click target. */}
      <mesh
        geometry={geometry}
        onPointerDown={(e: ThreeEvent<PointerEvent>) => {
          e.stopPropagation()
          const p = e.point
          // `p` is in world space. The surface is rotated, so translate back.
          onPick(p.x, p.z)
        }}
      >
        <meshStandardMaterial
          color="#1f1b3a"
          opacity={0.75}
          transparent
          metalness={0.1}
          roughness={0.8}
          side={THREE.DoubleSide}
        />
      </mesh>
      {/* Wireframe on top for that cross-hatched contour feel. */}
      <mesh geometry={geometry}>
        <meshBasicMaterial color="#a78bfa" wireframe transparent opacity={0.22} />
      </mesh>
    </group>
  )
}

function Floor() {
  // Faint grid shadow on the ground to ground the bowl in space.
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 0]}>
      <planeGeometry args={[2 * EXTENT, 2 * EXTENT]} />
      <meshBasicMaterial color="#0a0a0a" />
    </mesh>
  )
}

function Marble({ position }: { position: [number, number, number] }) {
  return (
    <mesh position={position} castShadow>
      <sphereGeometry args={[MARBLE_R, 32, 32]} />
      <meshStandardMaterial color="#fbbf24" emissive="#fbbf24" emissiveIntensity={0.4} />
    </mesh>
  )
}

function Trail({ points }: { points: THREE.Vector3[] }) {
  if (points.length < 2) return null
  return <Line points={points} color="#fbbf24" lineWidth={2} transparent opacity={0.8} />
}

interface DescentState {
  running: boolean
  x: number
  z: number
  trail: THREE.Vector3[]
}

function DescentSim({
  lr,
  state,
  setState,
}: {
  lr: number
  state: DescentState
  setState: (updater: (s: DescentState) => DescentState) => void
}) {
  const tickRef = useRef(0)
  useFrame((_, delta) => {
    if (!state.running) return
    tickRef.current += delta
    // Run a step every ~70ms — slow enough to watch, fast enough to feel live.
    if (tickRef.current < 0.07) return
    tickRef.current = 0

    setState((s) => {
      if (!s.running) return s
      const [gx, gz] = grad(s.x, s.z)
      const nx = s.x - lr * gx
      const nz = s.z - lr * gz
      const next = new THREE.Vector3(nx, f(nx, nz) + MARBLE_R, nz)
      const newTrail = [...s.trail, next]
      const magnitude = Math.hypot(gx, gz)
      const blewUp = Math.abs(nx) > 8 || Math.abs(nz) > 8
      const converged = magnitude < 0.002
      return {
        ...s,
        x: nx,
        z: nz,
        trail: newTrail.slice(-400),
        running: !blewUp && !converged,
      }
    })
  })
  return null
}

function HoverMarker({
  onHover,
}: {
  onHover: (p: THREE.Vector3 | null) => void
}) {
  const { camera, mouse, raycaster, scene } = useThree()
  useFrame(() => {
    raycaster.setFromCamera(mouse, camera)
    const intersects = raycaster.intersectObjects(scene.children, true)
    if (intersects.length > 0) {
      const p = intersects[0].point
      if (Math.abs(p.x) <= EXTENT && Math.abs(p.z) <= EXTENT) {
        onHover(p)
        return
      }
    }
    onHover(null)
  })
  return null
}

export default function LossSurface3D() {
  const [lr, setLr] = useState(0.15)
  const [state, setState] = useState<DescentState>({
    running: false,
    x: 2.3,
    z: 2.1,
    trail: [new THREE.Vector3(2.3, f(2.3, 2.1) + MARBLE_R, 2.1)],
  })
  const [hover, setHover] = useState<THREE.Vector3 | null>(null)

  const reset = () => {
    setState({
      running: false,
      x: 2.3,
      z: 2.1,
      trail: [new THREE.Vector3(2.3, f(2.3, 2.1) + MARBLE_R, 2.1)],
    })
  }

  const release = () => setState((s) => ({ ...s, running: true }))

  const placeMarble = (x: number, z: number) => {
    // Clamp just inside the surface.
    const cx = Math.max(-EXTENT + 0.1, Math.min(EXTENT - 0.1, x))
    const cz = Math.max(-EXTENT + 0.1, Math.min(EXTENT - 0.1, z))
    setState({
      running: false,
      x: cx,
      z: cz,
      trail: [new THREE.Vector3(cx, f(cx, cz) + MARBLE_R, cz)],
    })
  }

  const marbleY = f(state.x, state.z) + MARBLE_R
  const height = f(state.x, state.z)

  return (
    <WidgetFrame
      widgetName="LossSurface3D"
      label="the loss landscape"
      right={
        <>
          <span>f(x,z) = 0.2·(x² + z²)</span>
          <span className="text-dark-text-disabled">·</span>
          <span>drag to orbit</span>
        </>
      }
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="α"
            value={lr}
            min={0.01}
            max={0.9}
            step={0.01}
            onChange={setLr}
            accent="accent-term-purple"
          />
          <div className="flex items-center gap-2">
            <Button onClick={release} variant="primary" disabled={state.running}>
              <Play className="w-3 h-3 inline -mt-px mr-1" /> release
            </Button>
            <Button onClick={reset} disabled={state.running}>
              <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> reset
            </Button>
          </div>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="x" value={state.x.toFixed(2)} />
            <Readout label="z" value={state.z.toFixed(2)} />
            <Readout
              label="loss"
              value={height.toFixed(3)}
              accent="text-term-amber"
            />
          </div>
        </div>
      }
    >
      <Canvas
        shadows
        camera={{ position: [4.5, 3.8, 4.5], fov: 40 }}
        gl={{ antialias: true, alpha: true }}
      >
        <color attach="background" args={['#0a0a0a']} />
        <Suspense fallback={null}>
          <ambientLight intensity={0.4} />
          <directionalLight
            position={[5, 8, 3]}
            intensity={0.9}
            color="#ffffff"
            castShadow
          />
          <pointLight position={[-3, 4, -3]} intensity={0.3} color="#a78bfa" />

          <Floor />
          <Paraboloid onPick={placeMarble} />
          <Trail points={state.trail} />
          <Marble position={[state.x, marbleY, state.z]} />

          {hover && !state.running && (
            <mesh position={[hover.x, hover.y + 0.03, hover.z]}>
              <ringGeometry args={[MARBLE_R * 1.4, MARBLE_R * 1.8, 24]} />
              <meshBasicMaterial color="#a78bfa" side={THREE.DoubleSide} />
            </mesh>
          )}

          <DescentSim lr={lr} state={state} setState={setState} />
          <HoverMarker onHover={setHover} />
          <OrbitControls
            enablePan={false}
            minDistance={3.5}
            maxDistance={12}
            maxPolarAngle={Math.PI / 2 - 0.1}
          />
        </Suspense>
      </Canvas>

      {/* Gentle hint overlay — fades when user starts interacting. */}
      {!state.running && state.trail.length === 1 && (
        <div className="absolute top-3 left-3 flex items-center gap-1.5 text-[11px] font-mono text-dark-text-muted pointer-events-none">
          <MousePointer2 className="w-3 h-3" />
          <span>click the surface to place the marble</span>
        </div>
      )}
    </WidgetFrame>
  )
}
