import { OrbitControls } from '@react-three/drei'
import { Canvas, useThree, type ThreeEvent } from '@react-three/fiber'
import { useEffect, useMemo, useRef, type ReactNode } from 'react'
import * as THREE from 'three'

/** The point under the pointer, and where the pointer is in viewport (client) pixels. */
export interface PointHit {
  index: number
  x: number
  y: number
}

interface Props {
  geometry: THREE.BufferGeometry
  /** Multiplies the automatic point size, which follows the Scene's extent. */
  pointScale?: number
  /**
   * Called with the nearest point under a mouse pointer, or under a tap on a touch screen,
   * and with null when the pointer leaves the points or a tap misses them. Leave it unset
   * when nothing needs it: picking tests every point on each pointer move.
   */
  onPointHover?: (hit: PointHit | null) => void
  children?: ReactNode
}

/** A tap that moves further than this, in pixels, is a drag and picks nothing. */
const TAP_SLOP_PX = 6

/** Keep the raycaster's pick radius for points at the size points are drawn. */
function PointPickRadius({ radius }: { radius: number }) {
  const raycaster = useThree((s) => s.raycaster)
  useEffect(() => {
    raycaster.params.Points = { ...raycaster.params.Points, threshold: radius }
  }, [raycaster, radius])
  return null
}

/** Render a coloured point cloud with orbit/pan/zoom controls, framed to fit. */
export function PointCloudCanvas({ geometry, pointScale = 1, onPointHover, children }: Props) {
  const { center, radius } = useMemo(() => {
    geometry.computeBoundingSphere()
    const sphere = geometry.boundingSphere ?? new THREE.Sphere(new THREE.Vector3(), 1)
    return { center: sphere.center.clone(), radius: Math.max(sphere.radius, 0.1) }
  }, [geometry])

  const pointSize = Math.max(radius / 400, 0.004) * pointScale
  const lastPointerType = useRef<string>('mouse')

  const hitOf = (e: ThreeEvent<MouseEvent | PointerEvent>): PointHit | null =>
    e.index === undefined ? null : { index: e.index, x: e.nativeEvent.clientX, y: e.nativeEvent.clientY }

  const pickHandlers = onPointHover && {
    onPointerDown: (e: ThreeEvent<PointerEvent>) => {
      lastPointerType.current = e.nativeEvent.pointerType
    },
    // r3f calls a handler once per point under the ray, nearest first; stopping after the
    // first keeps the nearest point instead of letting a farther one overwrite it.
    onPointerMove: (e: ThreeEvent<PointerEvent>) => {
      e.stopPropagation()
      if (e.nativeEvent.pointerType !== 'touch') onPointHover(hitOf(e))
    },
    onPointerOut: (e: ThreeEvent<PointerEvent>) => {
      if (e.nativeEvent.pointerType !== 'touch') onPointHover(null)
    },
    onClick: (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation()
      if (lastPointerType.current === 'touch' && e.delta <= TAP_SLOP_PX) onPointHover(hitOf(e))
    },
  }

  return (
    <Canvas
      camera={{
        position: [center.x, center.y - radius * 1.6, center.z - radius * 1.2],
        up: [0, -1, 0],
        near: radius / 1000,
        far: radius * 20,
      }}
      onPointerMissed={onPointHover && (() => onPointHover(null))}
    >
      <color attach="background" args={['#111114']} />
      <PointPickRadius radius={pointSize / 2} />
      <points geometry={geometry} {...pickHandlers}>
        <pointsMaterial size={pointSize} vertexColors sizeAttenuation />
      </points>
      {children}
      <OrbitControls target={center} makeDefault />
    </Canvas>
  )
}
