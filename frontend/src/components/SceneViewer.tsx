import { Html } from '@react-three/drei'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import {
  getInstanceIds,
  getInstances,
  getOptions,
  getPointCloud,
  queryScene,
  type BackendConfig,
  type InstanceInfo,
  type QueryResult,
} from '../api'
import { heatmapColor, heatmapPositions, highlightColors, namesColors } from '../highlight'
import { PointCloudCanvas } from './PointCloudCanvas'

interface Props {
  config: BackendConfig
  sceneId: string
  onBack: () => void
}

interface LoadedScene {
  geometry: THREE.BufferGeometry
  baseColors: Float32Array
  instanceIds: Int32Array
  instances: Map<number, InstanceInfo>
}

const POINT_SCALE_KEY = 'scenetiq.pointScale'
const POINT_SCALE_MIN = 0.25
const POINT_SCALE_MAX = 4

/** The remembered point size; storage can be missing or throw, so fall back to 1. */
function readPointScale(): number {
  try {
    const value = Number(window.localStorage.getItem(POINT_SCALE_KEY))
    return value >= POINT_SCALE_MIN && value <= POINT_SCALE_MAX ? value : 1
  } catch {
    return 1
  }
}

function writePointScale(value: number): void {
  try {
    window.localStorage.setItem(POINT_SCALE_KEY, String(value))
  } catch {
    /* not remembered, which is fine */
  }
}

/** One line saying what of a Query was not taken into account, or null when it all was. */
function queryNotice(result: QueryResult): string | null {
  if (result.parse_failed) {
    return 'The Query could not be read for Relations, so it was matched as plain text.'
  }
  const parts = result.unsupported.map((p) => `"${p}"`)
  if (result.relations_stated && !result.relations_applied) {
    parts.push('its Relations, which no set of Instances satisfies')
  }
  return parts.length ? `Not taken into account: ${parts.join('; ')}.` : null
}

/** A linear colour, as the points use, written as the sRGB CSS colour that looks the same. */
function cssColor([r, g, b]: [number, number, number]): string {
  return new THREE.Color().setRGB(r, g, b, THREE.LinearSRGBColorSpace).getStyle()
}

/** How long the K slider must rest before the shown Query re-runs, so a drag runs one Query. */
const RERUN_DELAY_MS = 400

/** Load a Scene, fill the window with its point cloud, and highlight the Matches of a Query. */
export function SceneViewer({ config, sceneId, onBack }: Props) {
  const viewerRef = useRef<HTMLDivElement>(null)
  const [fullscreen, setFullscreen] = useState(false)
  const [scene, setScene] = useState<LoadedScene | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [text, setText] = useState('')
  const [result, setResult] = useState<QueryResult | null>(null)
  const [querying, setQuerying] = useState(false)
  const [queryError, setQueryError] = useState<string | null>(null)
  const [showNames, setShowNames] = useState(false)
  const [pointScale, setPointScale] = useState(readPointScale)
  const [topK, setTopK] = useState<number | null>(null)
  const [maxTopK, setMaxTopK] = useState<number | null>(null)
  // The text of the result shown, which a new K re-runs.
  const [shownText, setShownText] = useState<string | null>(null)

  useEffect(() => {
    getOptions(config)
      .then((o) => {
        setTopK(o.query_top_k)
        setMaxTopK(o.query_max_top_k)
      })
      .catch((err) => setQueryError(`Cannot read the backend's options: ${err}`))
  }, [config])

  useEffect(() => {
    let cancelled = false
    setScene(null)
    setError(null)
    setResult(null)
    setShowNames(false)
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

  // Follow the browser's state, which Esc can also change.
  useEffect(() => {
    const sync = () => setFullscreen(document.fullscreenElement !== null)
    document.addEventListener('fullscreenchange', sync)
    return () => document.removeEventListener('fullscreenchange', sync)
  }, [])

  const named = useMemo(
    () =>
      new Set(
        [...(scene?.instances.values() ?? [])].filter((i) => i.name).map((i) => i.instance_id),
      ),
    [scene],
  )

  // Recolour points whenever the Query result or the view mode changes; a Query result wins.
  useEffect(() => {
    if (!scene) return
    const colors =
      showNames && !result
        ? namesColors(scene.instanceIds, named)
        : highlightColors(scene.baseColors, scene.instanceIds, result?.matches ?? [])
    scene.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  }, [scene, result, showNames, named])

  // The Names view labels every named Instance at its centroid.
  const labels = useMemo(() => {
    if (!scene || !showNames || result) return []
    return [...scene.instances.values()]
      .filter((i) => i.name)
      .map((i) => ({ id: i.instance_id, name: i.name, at: new THREE.Vector3(...i.centroid) }))
  }, [scene, showNames, result])

  function toggleFullscreen() {
    const change = fullscreen
      ? document.exitFullscreen()
      : viewerRef.current?.requestFullscreen()
    change?.catch((err) => setError(String(err)))
  }

  function changePointScale(value: number) {
    setPointScale(value)
    writePointScale(value)
  }

  // Only the latest Query may show its answer; an older one still in flight is dropped.
  const latestQuery = useRef(0)
  const rerunTimer = useRef<number | undefined>(undefined)
  useEffect(() => () => window.clearTimeout(rerunTimer.current), [])

  async function runQuery(query: string, k: number | null) {
    const id = ++latestQuery.current
    setQuerying(true)
    setQueryError(null)
    try {
      const answer = await queryScene(config, sceneId, query, k ?? undefined)
      if (id !== latestQuery.current) return
      setResult(answer)
      setShownText(query)
    } catch (err) {
      if (id === latestQuery.current) setQueryError(String(err))
    } finally {
      if (id === latestQuery.current) setQuerying(false)
    }
  }

  function submit(e: React.FormEvent) {
    e.preventDefault()
    if (text.trim()) runQuery(text.trim(), topK)
  }

  function changeTopK(k: number) {
    setTopK(k)
    window.clearTimeout(rerunTimer.current)
    if (shownText) rerunTimer.current = window.setTimeout(() => runQuery(shownText, k), RERUN_DELAY_MS)
  }

  function clear() {
    window.clearTimeout(rerunTimer.current)
    latestQuery.current += 1
    setQuerying(false)
    setResult(null)
    setShownText(null)
  }

  const notice = result && queryNotice(result)
  const positions = result ? heatmapPositions(result.matches) : []

  return (
    <div className="viewer" ref={viewerRef}>
      {scene && (
        <div className="canvas-wrap">
          <PointCloudCanvas geometry={scene.geometry} pointScale={pointScale}>
            {labels.map((l) => (
              <Html key={l.id} position={l.at} center>
                <span className="instance-label">{l.name}</span>
              </Html>
            ))}
          </PointCloudCanvas>
        </div>
      )}
      <div className="viewer-overlay">
        <div className="viewer-toolbar">
          <button type="button" onClick={onBack}>
            ← Scenes
          </button>
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
              <button type="button" onClick={clear}>
                Clear
              </button>
            )}
          </form>
          <button
            type="button"
            aria-pressed={showNames}
            disabled={!scene || result !== null}
            title={result ? 'Clear the Query to view Names' : undefined}
            onClick={() => setShowNames((shown) => !shown)}
          >
            {showNames ? 'Original colours' : 'Names'}
          </button>
          {topK !== null && maxTopK !== null && (
            <label className="slider" title="How many Matches a Query returns">
              Matches {topK}
              <input
                type="range"
                min={1}
                max={maxTopK}
                step={1}
                value={topK}
                onChange={(e) => changeTopK(Number(e.target.value))}
              />
            </label>
          )}
          <label className="slider" title="Point size">
            Points
            <input
              type="range"
              min={POINT_SCALE_MIN}
              max={POINT_SCALE_MAX}
              step={0.25}
              value={pointScale}
              onChange={(e) => changePointScale(Number(e.target.value))}
            />
          </label>
          {document.fullscreenEnabled && (
            <button type="button" onClick={toggleFullscreen}>
              {fullscreen ? 'Exit full screen' : 'Full screen'}
            </button>
          )}
        </div>
        {(error || queryError || !scene || result) && (
          <div className="viewer-panel">
            {error && <p className="error">{error}</p>}
            {!scene && !error && <p className="muted">Loading Reconstruction…</p>}
            {queryError && <p className="error">{queryError}</p>}
            {result && (
              <div className="matches">
                {notice && <p className="muted">{notice}</p>}
                {result.matches.length === 0 && (
                  <p className="muted">This Scene has no Instances.</p>
                )}
                <ol>
                  {result.matches.map((m, rank) => (
                    <li key={m.instance_id}>
                      <span
                        className="swatch"
                        style={{ background: cssColor(heatmapColor(positions[rank])) }}
                      />
                      {scene?.instances.get(m.instance_id)?.name ?? `Instance ${m.instance_id}`}
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
