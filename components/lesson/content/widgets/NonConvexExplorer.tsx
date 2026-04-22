'use client'

import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber'
import { OrbitControls, Line } from '@react-three/drei'
import { Suspense, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import WidgetFrame, { Slider, Button, Readout } from './WidgetFrame'
import { RotateCcw, MousePointer2 } from 'lucide-react'

// A deliberately bumpy surface so the marble can fall into different basins
// depending on where it starts. Two cosine ripples + a gentle bowl keeps the
// whole thing bounded but riddled with local minima.
//
//   f(x, z) = 0.06·(x² + z²) + 0.9·[cos(1.1x) + cos(1.1z)]
//
//   ∂f/∂x = 0.12·x − 0.99·sin(1.1x)
//   ∂f/∂z = 0.12·z − 0.99·sin(1.1z)

const EXTENT = 4.5
const MARBLE_R = 0.15

function f(x: number, z: number) {
  return 0.06 * (x * x + z * z) + 0.9 * (Math.cos(1.1 * x) + Math.cos(1.1 * z))
}
function grad(x: number, z: number): [number, number] {
  return [0.12 * x - 0.99 * Math.sin(1.1 * x), 0.12 * z - 0.99 * Math.sin(1.1 * z)]
}

function Surface({ onPick }: { onPick: (x: number, z: number) => void }) {
  const geometry = useMemo(() => {
    const seg = 100
    const geo = new THREE.PlaneGeometry(2 * EXTENT, 2 * EXTENT, seg, seg)
    const pos = geo.attributes.position
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i)
      const y = pos.getY(i)
      pos.setZ(i, f(x, y))
    }
    geo.computeVertexNormals()
    return geo
  }, [])

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      <mesh
        geometry={geometry}
        onPointerDown={(e: ThreeEvent<PointerEvent>) => {
          e.stopPropagation()
          onPick(e.point.x, e.point.z)
        }}
      >
        <meshStandardMaterial
          color="#231a3a"
          opacity={0.8}
          transparent
          metalness={0.15}
          roughness={0.75}
          side={THREE.DoubleSide}
        />
      </mesh>
      <mesh geometry={geometry}>
        <meshBasicMaterial color="#a78bfa" wireframe transparent opacity={0.18} />
      </mesh>
    </group>
  )
}

function Marble({
  position,
  color,
}: {
  position: [number, number, number]
  color: string
}) {
  return (
    <mesh position={position}>
      <sphereGeometry args={[MARBLE_R, 24, 24]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.45} />
    </mesh>
  )
}

function Trail({ points, color }: { points: THREE.Vector3[]; color: string }) {
  if (points.length < 2) return null
  return <Line points={points} color={color} lineWidth={2} transparent opacity={0.85} />
}

interface Run {
  id: number
  color: string
  x: number
  z: number
  trail: THREE.Vector3[]
  running: boolean
  stuck: boolean
}

// Palette of distinct hues for each dropped marble. Cycles if you drop more.
const COLORS = ['#fbbf24', '#4ade80', '#f472b6', '#60a5fa', '#f87171', '#c084fc']

function Simulator({
  lr,
  runs,
  setRuns,
}: {
  lr: number
  runs: Run[]
  setRuns: React.Dispatch<React.SetStateAction<Run[]>>
}) {
  const tickRef = useRef(0)
  useFrame((_, delta) => {
    tickRef.current += delta
    if (tickRef.current < 0.06) return
    tickRef.current = 0
    setRuns((prev) =>
      prev.map((r) => {
        if (!r.running) return r
        const [gx, gz] = grad(r.x, r.z)
        const nx = r.x - lr * gx
        const nz = r.z - lr * gz
        const mag = Math.hypot(gx, gz)
        const outOfBounds = Math.abs(nx) > EXTENT || Math.abs(nz) > EXTENT
        const settled = mag < 0.003
        const nextTrail = [...r.trail, new THREE.Vector3(nx, f(nx, nz) + MARBLE_R, nz)]
        return {
          ...r,
          x: nx,
          z: nz,
          trail: nextTrail.slice(-400),
          running: !settled && !outOfBounds,
          stuck: settled || outOfBounds,
        }
      }),
    )
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
    const hits = raycaster.intersectObjects(scene.children, true)
    if (hits.length > 0) {
      const p = hits[0].point
      if (Math.abs(p.x) <= EXTENT && Math.abs(p.z) <= EXTENT) {
        onHover(p)
        return
      }
    }
    onHover(null)
  })
  return null
}

export default function NonConvexExplorer() {
  const [lr, setLr] = useState(0.18)
  const [runs, setRuns] = useState<Run[]>([])
  const [hover, setHover] = useState<THREE.Vector3 | null>(null)
  const counterRef = useRef(0)

  const dropMarble = (x: number, z: number) => {
    const cx = Math.max(-EXTENT + 0.1, Math.min(EXTENT - 0.1, x))
    const cz = Math.max(-EXTENT + 0.1, Math.min(EXTENT - 0.1, z))
    const color = COLORS[counterRef.current % COLORS.length]
    counterRef.current += 1
    setRuns((prev) => [
      ...prev,
      {
        id: counterRef.current,
        color,
        x: cx,
        z: cz,
        trail: [new THREE.Vector3(cx, f(cx, cz) + MARBLE_R, cz)],
        running: true,
        stuck: false,
      },
    ])
  }

  const clear = () => {
    setRuns([])
    counterRef.current = 0
  }

  // Count distinct final resting spots to expose the "same algorithm, different
  // answer" point. Cluster by rounding to a grid cell.
  const uniqueMinima = useMemo(() => {
    const cells = new Set<string>()
    runs.forEach((r) => {
      if (!r.stuck) return
      cells.add(`${Math.round(r.x * 2)}_${Math.round(r.z * 2)}`)
    })
    return cells.size
  }, [runs])

  return (
    <WidgetFrame
      widgetName="NonConvexExplorer"
      label="local minima — same algorithm, different answer"
      right={
        <>
          <span>f(x,z) = 0.06(x²+z²) + 0.9(cos 1.1x + cos 1.1z)</span>
          <span className="text-dark-text-disabled">·</span>
          <span>click anywhere to drop</span>
        </>
      }
      controls={
        <div className="flex flex-wrap items-center gap-4">
          <Slider
            label="α"
            value={lr}
            min={0.02}
            max={0.5}
            step={0.01}
            onChange={setLr}
            accent="accent-term-purple"
          />
          <Button onClick={clear} disabled={runs.length === 0}>
            <RotateCcw className="w-3 h-3 inline -mt-px mr-1" /> clear
          </Button>
          <div className="flex items-center gap-4 ml-auto">
            <Readout label="drops" value={String(runs.length)} />
            <Readout
              label="distinct minima"
              value={String(uniqueMinima)}
              accent="text-term-amber"
            />
          </div>
        </div>
      }
    >
      <Canvas camera={{ position: [6, 5.5, 6], fov: 40 }} gl={{ antialias: true, alpha: true }}>
        <color attach="background" args={['#0a0a0a']} />
        <Suspense fallback={null}>
          <ambientLight intensity={0.45} />
          <directionalLight position={[5, 8, 3]} intensity={0.9} color="#ffffff" />
          <pointLight position={[-4, 5, -4]} intensity={0.35} color="#a78bfa" />

          <Surface onPick={dropMarble} />

          {runs.map((r) => (
            <group key={r.id}>
              <Trail points={r.trail} color={r.color} />
              <Marble position={[r.x, f(r.x, r.z) + MARBLE_R, r.z]} color={r.color} />
            </group>
          ))}

          {hover && (
            <mesh position={[hover.x, hover.y + 0.03, hover.z]}>
              <ringGeometry args={[MARBLE_R * 1.4, MARBLE_R * 1.8, 24]} />
              <meshBasicMaterial color="#a78bfa" side={THREE.DoubleSide} />
            </mesh>
          )}

          <Simulator lr={lr} runs={runs} setRuns={setRuns} />
          <HoverMarker onHover={setHover} />
          <OrbitControls
            enablePan={false}
            minDistance={4.5}
            maxDistance={14}
            maxPolarAngle={Math.PI / 2 - 0.1}
          />
        </Suspense>
      </Canvas>

      {runs.length === 0 && (
        <div className="absolute top-3 left-3 flex items-center gap-1.5 text-[11px] font-mono text-dark-text-muted pointer-events-none">
          <MousePointer2 className="w-3 h-3" />
          <span>drop marbles on different spots — watch them land in different basins</span>
        </div>
      )}
    </WidgetFrame>
  )
}
