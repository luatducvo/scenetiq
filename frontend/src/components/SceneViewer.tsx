import { Html } from '@react-three/drei'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import {
  getInstanceIds,
  getInstances,
  getOptions,
  getPointCloud,
  listScenes,
  queryScene,
  type BackendConfig,
  type InstanceInfo,
  type QueryResult,
  type SceneSummary,
} from '../api'
import { highlightColors, namesColors } from '../highlight'
import { PointCloudCanvas } from './PointCloudCanvas'
import markSvg from '../assets/icons/mark.svg'
import iconArrowLeftSvg from '../assets/icons/icon-arrow-left.svg'
import iconSearchSvg from '../assets/icons/icon-search.svg'
import iconSlidersSvg from '../assets/icons/icon-sliders.svg'
import iconMaximizeSvg from '../assets/icons/icon-maximize.svg'
import iconXSvg from '../assets/icons/icon-x.svg'
import iconFocusSvg from '../assets/icons/icon-focus.svg'

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
    /* ignore */
  }
}

const RANK_COLORS = ['#ff5a36', '#ff9f1c', '#ffd166']
function getRankColor(rank: number): string {
  return RANK_COLORS[rank] || '#ffd166'
}

const SUGGESTIONS = ['red chair', 'something to sit on', 'the lamp near the sofa']
const RERUN_DELAY_MS = 400

export function SceneViewer({ config, sceneId, onBack }: Props) {
  const viewerRef = useRef<HTMLDivElement>(null)
  const [fullscreen, setFullscreen] = useState(false)
  const [sceneSummary, setSceneSummary] = useState<SceneSummary | null>(null)
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
  const [shownText, setShownText] = useState<string | null>(null)
  const [showDisplayOptions, setShowDisplayOptions] = useState(false)
  const [focusedInstanceId, setFocusedInstanceId] = useState<number | null>(null)

  useEffect(() => {
    listScenes(config)
      .then((all) => {
        const found = all.find((s) => s.scene_id === sceneId)
        if (found) setSceneSummary(found)
      })
      .catch(() => {})

    getOptions(config)
      .then((o) => {
        setTopK(o.query_top_k)
        setMaxTopK(o.query_max_top_k)
      })
      .catch((err) => setQueryError(`Cannot read the backend's options: ${err}`))
  }, [config, sceneId])

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

  useEffect(() => {
    if (!scene) return
    const colors =
      showNames && !result
        ? namesColors(scene.instanceIds, named)
        : highlightColors(scene.baseColors, scene.instanceIds, result?.matches ?? [])
    scene.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  }, [scene, result, showNames, named])

  const labels = useMemo(() => {
    if (!scene || !showNames || result) return []
    return [...scene.instances.values()]
      .filter((i) => i.name)
      .map((i) => ({ id: i.instance_id, name: i.name, at: new THREE.Vector3(...i.centroid) }))
  }, [scene, showNames, result])

  const matchPins = useMemo(() => {
    if (!scene || !result || result.matches.length === 0) return []
    return result.matches
      .map((m, rank) => {
        const inst = scene.instances.get(m.instance_id)
        if (!inst || !inst.centroid) return null
        return {
          id: m.instance_id,
          rank: rank + 1,
          name: inst.name || `Object #${m.instance_id}`,
          color: getRankColor(rank),
          at: new THREE.Vector3(...inst.centroid),
        }
      })
      .filter(Boolean) as Array<{
      id: number
      rank: number
      name: string
      color: string
      at: THREE.Vector3
    }>
  }, [scene, result])

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

  function handleSearchSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (text.trim()) runQuery(text.trim(), topK)
  }

  function applySuggestion(s: string) {
    setText(s)
    runQuery(s, topK)
  }

  function clearSearch() {
    window.clearTimeout(rerunTimer.current)
    latestQuery.current += 1
    setQuerying(false)
    setResult(null)
    setShownText(null)
    setText('')
    setFocusedInstanceId(null)
  }

  function changeTopK(k: number) {
    setTopK(k)
    window.clearTimeout(rerunTimer.current)
    if (shownText) rerunTimer.current = window.setTimeout(() => runQuery(shownText, k), RERUN_DELAY_MS)
  }

  const pointSizeLabel =
    pointScale <= 0.6 ? 'Small' : pointScale <= 1.4 ? 'Medium' : 'Large'

  return (
    <div className="scene-viewer-root" ref={viewerRef}>
      {scene && (
        <div className="viewer-canvas-wrap">
          <PointCloudCanvas geometry={scene.geometry} pointScale={pointScale}>
            {labels.map((l) => (
              <Html key={l.id} position={l.at} center>
                <div className="scene-label-tag">{l.name}</div>
              </Html>
            ))}
            {matchPins.map((pin) => (
              <Html key={pin.id} position={pin.at} center>
                <div
                  className={`spatial-pin ${focusedInstanceId === pin.id ? 'spatial-pin-focused' : ''}`}
                  onClick={() => setFocusedInstanceId(pin.id)}
                >
                  <span className="spatial-pin-rank" style={{ background: pin.color }}>
                    {pin.rank}
                  </span>
                  <span className="spatial-pin-name">{pin.name}</span>
                </div>
              </Html>
            ))}
          </PointCloudCanvas>
        </div>
      )}

      {/* Floating UI Overlays */}
      <div className="viewer-ui-overlay">
        {/* Top-left: Scene Chip matching Figma node 10:204 */}
        <div className="scene-chip" onClick={onBack} role="button" tabIndex={0}>
          <div className="scene-chip-brand">
            <img src={markSvg} alt="SceneTIQ" width={28} height={28} />
          </div>
          <div className="scene-chip-divider" />
          <div className="scene-chip-back-icon">
            <img src={iconArrowLeftSvg} alt="" width={18} height={18} />
          </div>
          <div className="scene-chip-text">
            <span className="scene-chip-title">{sceneSummary?.name || 'Scene'}</span>
            <span className="scene-chip-objects">
              {sceneSummary?.instance_count != null
                ? `${sceneSummary.instance_count} objects`
                : `${scene?.instances.size ?? 0} objects`}
            </span>
          </div>
        </div>

        {/* Top-center: Search Bar matching Figma node 10:153 / 10:161 */}
        <div className="search-bar-container">
          <form className="viewer-search-bar" onSubmit={handleSearchSubmit}>
            <div className="search-bar-icon">
              <img src={iconSearchSvg} alt="" width={20} height={20} />
            </div>
            <input
              type="text"
              className="search-bar-input"
              placeholder="Describe an object, like “red chair”"
              value={text}
              onChange={(e) => setText(e.target.value)}
              disabled={!scene}
            />
            {text && (
              <button
                type="button"
                className="search-clear-btn"
                onClick={clearSearch}
                title="Clear query"
              >
                <img src={iconXSvg} alt="" width={16} height={16} />
              </button>
            )}
            <button
              type="submit"
              className="search-find-btn"
              disabled={!scene || querying || !text.trim()}
            >
              {querying ? 'Finding…' : 'Find'}
            </button>
          </form>

          {/* Suggestions chips below search bar matching Figma node 10:238 */}
          {!result && (
            <div className="suggestions-row">
              <span className="suggestions-try">Try</span>
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  type="button"
                  className="suggestion-chip"
                  onClick={() => applySuggestion(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Top-right: Tools buttons matching Figma node 10:225 */}
        <div className="viewer-tools">
          <button
            type="button"
            className={`tool-icon-btn ${showDisplayOptions ? 'tool-btn-active' : ''}`}
            onClick={() => setShowDisplayOptions(!showDisplayOptions)}
            title="Display options"
          >
            <img src={iconSlidersSvg} alt="" width={20} height={20} />
          </button>
          {document.fullscreenEnabled && (
            <button
              type="button"
              className="tool-icon-btn"
              onClick={toggleFullscreen}
              title={fullscreen ? 'Exit fullscreen' : 'Fullscreen'}
            >
              <img src={iconMaximizeSvg} alt="" width={20} height={20} />
            </button>
          )}
        </div>

        {/* Display Options panel matching Figma node 10:246 */}
        {showDisplayOptions && (
          <div className="display-options-panel">
            <h3 className="display-options-title">Display</h3>

            <div className="display-option-group">
              <div className="display-option-label-row">
                <span className="option-name">Point size</span>
                <span className="option-val">{pointSizeLabel}</span>
              </div>
              <input
                type="range"
                className="option-slider"
                min={POINT_SCALE_MIN}
                max={POINT_SCALE_MAX}
                step={0.25}
                value={pointScale}
                onChange={(e) => changePointScale(Number(e.target.value))}
              />
            </div>

            {topK !== null && maxTopK !== null && (
              <div className="display-option-group">
                <div className="display-option-label-row">
                  <span className="option-name">Results to show</span>
                  <span className="option-val">{topK}</span>
                </div>
                <input
                  type="range"
                  className="option-slider"
                  min={1}
                  max={maxTopK}
                  step={1}
                  value={topK}
                  onChange={(e) => changeTopK(Number(e.target.value))}
                />
              </div>
            )}

            <div className="display-option-toggle-row">
              <div className="toggle-text-wrap">
                <span className="toggle-title">Show object names</span>
                <span className="toggle-desc">Label every object in the scene</span>
              </div>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={showNames}
                  disabled={!scene || result !== null}
                  onChange={(e) => setShowNames(e.target.checked)}
                />
                <span className="switch-slider round" />
              </label>
            </div>
          </div>
        )}

        {/* Search Results Panel matching Figma node 10:327 */}
        {result && (
          <div className="results-panel">
            <div className="results-header">
              <h3 className="results-title">
                {result.matches.length} {result.matches.length === 1 ? 'match' : 'matches'}
              </h3>
              <p className="results-subtitle">for “{shownText}”, best first</p>
            </div>

            <div className="results-list">
              {result.matches.length === 0 && (
                <p className="muted results-empty">No objects found matching “{shownText}”.</p>
              )}
              {result.matches.map((m, rank) => {
                const inst = scene?.instances.get(m.instance_id)
                const name = inst?.name || `Object #${m.instance_id}`
                const detail = rank === 0 ? 'Best match' : `Score ${(m.score * 100).toFixed(0)}%`
                const rankColor = getRankColor(rank)
                const isFocused = focusedInstanceId === m.instance_id

                return (
                  <div
                    key={m.instance_id}
                    className={`match-row ${isFocused ? 'match-row-focused' : ''} ${
                      rank === 0 ? 'match-row-top' : ''
                    }`}
                    onClick={() => setFocusedInstanceId(m.instance_id)}
                  >
                    <div className="match-swatch" style={{ background: rankColor }}>
                      <span className="match-swatch-rank">{rank + 1}</span>
                    </div>
                    <div className="match-info">
                      <span className="match-name">{name}</span>
                      <span className="match-detail">{detail}</span>
                    </div>
                    <button
                      type="button"
                      className="match-focus-btn"
                      title="Focus in 3D"
                      onClick={(e) => {
                        e.stopPropagation()
                        setFocusedInstanceId(m.instance_id)
                      }}
                    >
                      <img src={iconFocusSvg} alt="" width={18} height={18} />
                    </button>
                  </div>
                )
              })}
            </div>

            <div className="results-footer">
              <button
                type="button"
                className="btn-show-more"
                onClick={() => {
                  if (topK && maxTopK) {
                    changeTopK(Math.min(maxTopK, topK + 3))
                  }
                }}
              >
                Show more results
              </button>
              <button type="button" className="btn btn-secondary btn-sm" onClick={clearSearch}>
                Clear
              </button>
            </div>
          </div>
        )}

        {/* Bottom-center: Controls Hint matching Figma node 10:270 */}
        <div className="controls-hint">
          <span>Drag to turn</span>
          <span>·</span>
          <span>Scroll to zoom</span>
          <span>·</span>
          <span>Right-drag to move</span>
        </div>

        {/* Global errors or loading indicator */}
        {error && <div className="viewer-error-banner">{error}</div>}
        {queryError && <div className="viewer-error-banner">{queryError}</div>}
        {!scene && !error && (
          <div className="viewer-loading-overlay">
            <span className="muted">Loading reconstruction…</span>
          </div>
        )}
      </div>
    </div>
  )
}
