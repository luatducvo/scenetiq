import { useEffect, useMemo, useState } from 'react'
import * as THREE from 'three'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import {
  getInstanceIds,
  getInstances,
  getPointCloud,
  queryScene,
  type BackendConfig,
  type InstanceInfo,
  type QueryResult,
} from '../api'
import { highlightColors, matchColor } from '../highlight'
import { PointCloudCanvas } from './PointCloudCanvas'

interface Props {
  config: BackendConfig
  sceneId: string
}

interface LoadedScene {
  geometry: THREE.BufferGeometry
  baseColors: Float32Array
  instanceIds: Int32Array
  instances: Map<number, InstanceInfo>
}

/** Load a Scene, show it as a point cloud, and highlight the Matches of a Query. */
export function SceneViewer({ config, sceneId }: Props) {
  const [scene, setScene] = useState<LoadedScene | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [text, setText] = useState('')
  const [result, setResult] = useState<QueryResult | null>(null)
  const [querying, setQuerying] = useState(false)
  const [queryError, setQueryError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setScene(null)
    setError(null)
    setResult(null)
    Promise.all([
      getPointCloud(config, sceneId),
      getInstanceIds(config, sceneId),
      getInstances(config, sceneId),
    ])
      .then(([ply, instanceIds, instances]) => {
        if (cancelled) return
        const geometry = new PLYLoader().parse(ply)
        const color = geometry.getAttribute('color') as THREE.BufferAttribute | undefined
        const count = geometry.getAttribute('position').count
        const baseColors = color
          ? new Float32Array(color.array as ArrayLike<number>)
          : new Float32Array(count * 3).fill(0.8)
        if (!color) geometry.setAttribute('color', new THREE.BufferAttribute(baseColors.slice(), 3))
        setScene({
          geometry,
          baseColors,
          instanceIds,
          instances: new Map(instances.map((i) => [i.instance_id, i])),
        })
      })
      .catch((err) => !cancelled && setError(String(err)))
    return () => {
      cancelled = true
    }
  }, [config, sceneId])

  useEffect(() => () => scene?.geometry.dispose(), [scene])

  // Recolour points whenever the Query result changes.
  useEffect(() => {
    if (!scene) return
    const colors = highlightColors(scene.baseColors, scene.instanceIds, result?.matches ?? [])
    scene.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  }, [scene, result])

  const boxes = useMemo(() => {
    if (!scene || !result) return []
    return result.matches.flatMap((m, rank) => {
      const inst = scene.instances.get(m.instance_id)
      if (!inst) return []
      const box = new THREE.Box3(
        new THREE.Vector3(...inst.bbox_min),
        new THREE.Vector3(...inst.bbox_max),
      )
      return [new THREE.Box3Helper(box, new THREE.Color(...matchColor(rank)))]
    })
  }, [scene, result])

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    if (!text.trim()) return
    setQuerying(true)
    setQueryError(null)
    try {
      setResult(await queryScene(config, sceneId, text.trim()))
    } catch (err) {
      setQueryError(String(err))
    } finally {
      setQuerying(false)
    }
  }

  return (
    <div className="viewer">
      <form className="query-bar" onSubmit={submit}>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder='Describe an object, e.g. "red chair"'
          disabled={!scene}
        />
        <button type="submit" disabled={!scene || querying || !text.trim()}>
          {querying ? 'Searching…' : 'Query'}
        </button>
        {result && (
          <button type="button" onClick={() => setResult(null)}>
            Clear
          </button>
        )}
      </form>
      {queryError && <p className="error">{queryError}</p>}
      {result && (
        <div className="matches">
          {result.matches.length === 0 && <p className="muted">This Scene has no Instances.</p>}
          {result.low_confidence && result.matches.length > 0 && (
            <p className="warning">
              No confident Match. Showing the closest Instance as a Low-confidence Match.
            </p>
          )}
          <ol>
            {result.matches.map((m, rank) => (
              <li key={m.instance_id}>
                <span className="swatch" style={{ background: `rgb(${matchColor(rank).map((c) => c * 255).join(',')})` }} />
                Instance {m.instance_id} — score {m.score.toFixed(3)}
                {result.low_confidence && <span className="badge">low confidence</span>}
              </li>
            ))}
          </ol>
        </div>
      )}
      {error && <p className="error">{error}</p>}
      {!scene && !error && <p className="muted">Loading Reconstruction…</p>}
      {scene && (
        <div className="canvas-wrap">
          <PointCloudCanvas geometry={scene.geometry}>
            {boxes.map((b, i) => (
              <primitive key={i} object={b} />
            ))}
          </PointCloudCanvas>
        </div>
      )}
    </div>
  )
}
