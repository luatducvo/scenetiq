import { useEffect, useState } from 'react'
import * as THREE from 'three'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import { getPointCloud, type BackendConfig } from '../api'
import { PointCloudCanvas } from './PointCloudCanvas'

interface Props {
  config: BackendConfig
  sceneId: string
}

/** Load a Scene's Reconstruction and show it as an interactive point cloud. */
export function SceneViewer({ config, sceneId }: Props) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setGeometry(null)
    setError(null)
    getPointCloud(config, sceneId)
      .then((buf) => {
        if (cancelled) return
        const geom = new PLYLoader().parse(buf)
        setGeometry(geom)
      })
      .catch((err) => !cancelled && setError(String(err)))
    return () => {
      cancelled = true
    }
  }, [config, sceneId])

  useEffect(() => () => geometry?.dispose(), [geometry])

  return (
    <div className="viewer">
      {error && <p className="error">{error}</p>}
      {!geometry && !error && <p className="muted">Loading point cloud…</p>}
      {geometry && (
        <div className="canvas-wrap">
          <PointCloudCanvas geometry={geometry} />
        </div>
      )}
    </div>
  )
}
