import { useState } from 'react'
import type { SceneSummary } from '../api'
import { StatusBadge } from './StatusBadge'
import pointCloudSvg from '../assets/icons/point-cloud-preview.svg'
import iconMoreSvg from '../assets/icons/icon-more.svg'
import iconLoaderSvg from '../assets/icons/icon-loader.svg'
import iconAlertSvg from '../assets/icons/icon-alert.svg'

interface SceneCardProps {
  scene: SceneSummary
  onOpen: (sceneId: string) => void
  onOpenJob: (jobId: string) => void
  onDelete: (scene: SceneSummary) => void
  onCancel: (scene: SceneSummary) => void
}

function formatRelativeTime(dateStr?: string | null): string {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  const now = new Date()
  const diffSec = Math.floor((now.getTime() - date.getTime()) / 1000)
  if (diffSec < 60) return 'just now'
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)} min ago`
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`
  const diffDays = Math.floor(diffSec / 86400)
  if (diffDays === 1) return 'yesterday'
  if (diffDays < 7) return `${diffDays} days ago`
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export function SceneCard({ scene, onOpen, onOpenJob, onDelete, onCancel }: SceneCardProps) {
  const [menuOpen, setMenuOpen] = useState(false)
  const isReady = scene.status === 'succeeded'
  const isProcessing = scene.status === 'running' || scene.status === 'queued'
  const isFailed = scene.status === 'failed' || scene.status === 'cancelled'

  function handleClick() {
    if (isReady) {
      onOpen(scene.scene_id)
    } else if (isProcessing) {
      onOpenJob(scene.job_id)
    }
  }

  // Meta line formatting
  const metaParts: string[] = []
  if (scene.instance_count != null) {
    metaParts.push(`${scene.instance_count} objects`)
  }
  const timeStr = formatRelativeTime(scene.created_at)
  if (timeStr) {
    metaParts.push(timeStr)
  }
  const metaText = metaParts.length > 0 ? metaParts.join(' · ') : 'Recently created'

  return (
    <div
      className={`scene-card ${isReady || isProcessing ? 'clickable' : ''}`}
      onClick={handleClick}
    >
      <div className={`scene-card-thumb ${!isReady ? 'thumb-subtle' : ''}`}>
        {isReady && (
          <img
            src={pointCloudSvg}
            alt=""
            className="scene-thumb-img"
            loading="lazy"
          />
        )}
        {isProcessing && (
          <div className="scene-thumb-status">
            <img src={iconLoaderSvg} alt="" className="thumb-status-icon spin" width={28} height={28} />
            <span className="thumb-status-text">Processing…</span>
          </div>
        )}
        {isFailed && (
          <div className="scene-thumb-status">
            <img src={iconAlertSvg} alt="" className="thumb-status-icon" width={28} height={28} />
            <span className="thumb-status-text">Couldn't build scene</span>
          </div>
        )}
      </div>

      <div className="scene-card-body">
        <div className="scene-card-header">
          <h3 className="scene-card-title">{scene.name}</h3>
          <div
            className="scene-card-more-wrap"
            onClick={(e) => {
              e.stopPropagation()
              setMenuOpen(!menuOpen)
            }}
          >
            <button className="scene-more-btn" title="More options" aria-label="More options">
              <img src={iconMoreSvg} alt="" width={20} height={20} />
            </button>
            {menuOpen && (
              <div
                className="scene-dropdown-menu"
                onMouseLeave={() => setMenuOpen(false)}
                onClick={(e) => e.stopPropagation()}
              >
                {isProcessing ? (
                  <button
                    className="dropdown-item danger"
                    onClick={() => {
                      setMenuOpen(false)
                      onCancel(scene)
                    }}
                  >
                    Cancel build
                  </button>
                ) : (
                  <button
                    className="dropdown-item danger"
                    onClick={() => {
                      setMenuOpen(false)
                      onDelete(scene)
                    }}
                  >
                    Delete scene
                  </button>
                )}
              </div>
            )}
          </div>
        </div>

        <p className="scene-card-meta">{metaText}</p>

        {isProcessing && (
          <div className="scene-card-progress">
            <div className="scene-card-progress-bar" style={{ width: '60%' }} />
          </div>
        )}

        <div className="scene-card-footer">
          <StatusBadge status={scene.status} />
        </div>
      </div>
    </div>
  )
}
