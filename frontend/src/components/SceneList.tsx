import { useCallback, useEffect, useState } from 'react'
import { deleteScene, listScenes, type BackendConfig, type SceneSummary } from '../api'

interface Props {
  config: BackendConfig
  onOpen: (sceneId: string) => void
  onOpenJob: (jobId: string) => void
}

/** List processed Scenes with open and delete actions. */
export function SceneList({ config, onOpen, onOpenJob }: Props) {
  const [scenes, setScenes] = useState<SceneSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)

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

  return (
    <div className="panel">
      <h2>Scenes</h2>
      {error && <p className="error">{error}</p>}
      {scenes === null && !error && <p className="muted">Loading…</p>}
      {scenes?.length === 0 && <p className="muted">No Scenes yet. Upload a Capture to start.</p>}
      {scenes && scenes.length > 0 && (
        <table className="scene-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Created</th>
              <th>Status</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {scenes.map((s) => (
              <tr key={s.scene_id}>
                <td>{s.name}</td>
                <td>{new Date(s.created_at).toLocaleString()}</td>
                <td className={`status-${s.status}`}>{s.status}</td>
                <td className="actions">
                  {s.status === 'succeeded' && (
                    <button className="primary" onClick={() => onOpen(s.scene_id)}>
                      Open
                    </button>
                  )}
                  {(s.status === 'queued' || s.status === 'running') && (
                    <button onClick={() => onOpenJob(s.job_id)}>Progress</button>
                  )}
                  {s.status !== 'queued' && s.status !== 'running' && (
                    <button className="danger" onClick={() => remove(s)}>
                      Delete
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <button onClick={refresh}>Refresh</button>
    </div>
  )
}
