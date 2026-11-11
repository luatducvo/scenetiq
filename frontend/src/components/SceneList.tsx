import { useCallback, useEffect, useState } from 'react'
import { cancelJob, deleteScene, listScenes, type BackendConfig, type SceneSummary } from '../api'
import { SceneCard } from './SceneCard'
import iconPlusLargeSvg from '../assets/icons/icon-plus-large.svg'
import iconSlidersSvg from '../assets/icons/icon-sliders.svg'

interface Props {
  config: BackendConfig
  onOpen: (sceneId: string) => void
  onOpenJob: (jobId: string) => void
  onNewScene: () => void
}

export function SceneList({ config, onOpen, onOpenJob, onNewScene }: Props) {
  const [scenes, setScenes] = useState<SceneSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [sortNewest, setSortNewest] = useState(true)

  const refresh = useCallback(() => {
    listScenes(config)
      .then((s) => {
        setScenes(s)
        setError(null)
      })
      .catch((err) => setError(String(err)))
  }, [config])

  useEffect(refresh, [refresh])

  async function remove(scene: SceneSummary) {
    if (!window.confirm(`Delete Scene "${scene.name}"? This cannot be undone.`)) return
    try {
      await deleteScene(config, scene.scene_id)
      refresh()
    } catch (err) {
      setError(String(err))
    }
  }

  async function cancel(scene: SceneSummary) {
    if (!window.confirm(`Cancel processing "${scene.name}"? Its photos will be deleted.`)) return
    try {
      await cancelJob(config, scene.job_id)
      refresh()
    } catch (err) {
      setError(String(err))
    }
  }

  const sortedScenes = scenes
    ? [...scenes].sort((a, b) => {
        const da = new Date(a.created_at).getTime()
        const db = new Date(b.created_at).getTime()
        return sortNewest ? db - da : da - db
      })
    : []

  return (
    <div className="scenes-page">
      <div className="scenes-header">
        <div className="scenes-title-group">
          <h1 className="scenes-title">Your scenes</h1>
          <p className="scenes-subtitle">Photograph a room, then find anything in it by describing it.</p>
        </div>
        <button
          className="btn btn-ghost sort-btn"
          onClick={() => setSortNewest(!sortNewest)}
          title="Toggle sort order"
        >
          <img src={iconSlidersSvg} alt="" width={16} height={16} />
          <span>{sortNewest ? 'Newest first' : 'Oldest first'}</span>
        </button>
      </div>

      {error && <div className="banner error-banner">{error}</div>}

      {scenes === null && !error && (
        <div className="loading-state">
          <span className="muted">Loading scenes…</span>
        </div>
      )}

      {scenes !== null && (
        <div className="scenes-grid">
          {sortedScenes.map((s) => (
            <SceneCard
              key={s.scene_id}
              scene={s}
              onOpen={onOpen}
              onOpenJob={onOpenJob}
              onDelete={remove}
              onCancel={cancel}
            />
          ))}

          {/* New scene tile matching Figma node 8:172 */}
          <div className="new-scene-tile" onClick={onNewScene} role="button" tabIndex={0}>
            <div className="new-scene-tile-icon-wrap">
              <img src={iconPlusLargeSvg} alt="" width={24} height={24} />
            </div>
            <span className="new-scene-tile-title">New scene</span>
            <span className="new-scene-tile-subtitle">Upload photos of one room</span>
          </div>
        </div>
      )}
    </div>
  )
}
