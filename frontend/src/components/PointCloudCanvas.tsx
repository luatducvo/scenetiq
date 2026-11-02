import { OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { useMemo, type ReactNode } from 'react'
import * as THREE from 'three'

interface Props {
  geometry: THREE.BufferGeometry
  children?: ReactNode
}

/** Render a coloured point cloud with orbit/pan/zoom controls, framed to fit. */
export function PointCloudCanvas({ geometry, children }: Props) {
  const { center, radius } = useMemo(() => {
    geometry.computeBoundingSphere()
    const sphere = geometry.boundingSphere ?? new THREE.Sphere(new THREE.Vector3(), 1)
    return { center: sphere.center.clone(), radius: Math.max(sphere.radius, 0.1) }
  }, [geometry])

  const pointSize = Math.max(radius / 400, 0.004)

  return (
    <Canvas
      camera={{
        position: [center.x, center.y - radius * 1.6, center.z - radius * 1.2],
        up: [0, -1, 0],
        near: radius / 1000,
        far: radius * 20,
      }}
    >
      <color attach="background" args={['#111114']} />
      <points geometry={geometry}>
        <pointsMaterial size={pointSize} vertexColors sizeAttenuation />
      </points>
      {children}
      <OrbitControls target={center} makeDefault />
    </Canvas>
  )
}
