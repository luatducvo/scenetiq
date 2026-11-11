import React, { useEffect, useRef, useState } from 'react'
import {
  ApiError,
  createScene,
  getOptions,
  type BackendConfig,
  type BackendOptions,
  type UploadProgress,
} from '../api'
import { formatBytes, preparePhotos, releasePhoto, type PreparedPhoto } from '../photos'
import iconArrowLeftSvg from '../assets/icons/icon-arrow-left.svg'
import iconUploadSvg from '../assets/icons/icon-upload.svg'
import iconXSvg from '../assets/icons/icon-x.svg'
import iconEyeSvg from '../assets/icons/icon-eye.svg'
import iconCheckSvg from '../assets/icons/icon-check.svg'

interface Props {
  config: BackendConfig
  onStarted: (jobId: string, sceneId: string) => void
  onCancel: () => void
}

const ACCEPTED = /\.(jpe?g|png)$/i

export function UploadForm({ config, onStarted, onCancel }: Props) {
  const [options, setOptions] = useState<BackendOptions | null>(null)
  const [optionsError, setOptionsError] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [photos, setPhotos] = useState<PreparedPhoto[]>([])
  const [preparing, setPreparing] = useState<{ done: number; total: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState<UploadProgress | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const selection = useRef(0)
  const shown = useRef<PreparedPhoto[]>([])

  useEffect(() => {
    getOptions(config)
      .then(setOptions)
      .catch((err) => setOptionsError(String(err)))
  }, [config])

  useEffect(() => {
    shown.current = photos
  }, [photos])
  useEffect(() => () => shown.current.forEach(releasePhoto), [])

  async function handleFiles(list: FileList | null) {
    if (!options) return
    const files = Array.from(list ?? [])
    const run = ++selection.current
    photos.forEach(releasePhoto)
    setPhotos([])
    setError(null)
    const images = files.filter((f) => ACCEPTED.test(f.name))
    const notImages = files.filter((f) => !ACCEPTED.test(f.name)).map((f) => f.name)
    if (notImages.length > 0) {
      setError(`Some files were skipped (not JPG/PNG): ${notImages.join(', ')}`)
    }
    if (images.length === 0) return
    setPreparing({ done: 0, total: images.length })
    const result = await preparePhotos(
      images,
      { maxSide: options.upload_max_side, quality: options.upload_jpeg_quality },
      (done, total) => run === selection.current && setPreparing({ done, total }),
    )
    if (run !== selection.current) {
      result.photos.forEach(releasePhoto)
      return
    }
    setPhotos(result.photos)
    if (result.failed.length > 0) {
      setError(`Could not decode ${result.failed.length} images`)
    }
    setPreparing(null)
  }

  function remove(photo: PreparedPhoto) {
    releasePhoto(photo)
    setPhotos((all) => all.filter((p) => p.id !== photo.id))
  }

  const count = photos.length
  const countError =
    options && count > 0 && (count < options.min_views || count > options.max_views)
      ? `A Capture needs ${options.min_views}–${options.max_views} photos; ${count} selected.`
      : null
  const before = photos.reduce((sum, p) => sum + p.originalBytes, 0)
  const after = photos.reduce((sum, p) => sum + p.file.size, 0)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (countError || count === 0 || preparing || busy) return
    setBusy(true)
    setError(null)
    setProgress(null)
    try {
      const files = photos.map((p) => p.file)
      const sceneName = name.trim() || 'Untitled scene'
      const { job_id, scene_id } = await createScene(config, sceneName, files, setProgress)
      onStarted(job_id, scene_id)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="new-scene-page">
      <div className="new-scene-container">
        <button className="btn btn-ghost back-btn" onClick={onCancel} type="button">
          <img src={iconArrowLeftSvg} alt="" width={18} height={18} />
          <span>Scenes</span>
        </button>

        <div className="new-scene-header">
          <h1 className="new-scene-title">New scene</h1>
          <p className="new-scene-subtitle">
            Walk around one room and take photos from many angles. We'll turn them into 3D.
          </p>
        </div>

        {optionsError && <div className="banner error-banner">{optionsError}</div>}
        {error && <div className="banner error-banner">{error}</div>}
        {countError && <div className="banner warning-banner">{countError}</div>}

        <form onSubmit={handleSubmit} className="new-scene-layout">
          {/* Left card: Upload form */}
          <div className="new-scene-card">
            <div className="form-field">
              <label htmlFor="scene-name" className="field-label">
                Scene name
              </label>
              <input
                id="scene-name"
                type="text"
                className="field-input"
                placeholder="e.g. Living room"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>

            <div
              className={`drop-zone ${isDragging ? 'drop-zone-active' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault()
                setIsDragging(true)
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                e.preventDefault()
                setIsDragging(false)
                handleFiles(e.dataTransfer.files)
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept="image/jpeg,image/png"
                style={{ display: 'none' }}
                onChange={(e) => handleFiles(e.target.files)}
              />
              <div className="drop-zone-icon-wrap">
                <img src={iconUploadSvg} alt="" width={24} height={24} />
              </div>
              <p className="drop-zone-prompt">
                <span>Drop photos here, or </span>
                <span className="drop-zone-browse">browse</span>
              </p>
              <p className="drop-zone-hint">JPG or PNG · we shrink them before upload</p>
            </div>

            {preparing && (
              <p className="muted preparing-text">
                Preparing photos: {preparing.done}/{preparing.total}…
              </p>
            )}

            {count > 0 && (
              <div className="photos-summary">
                <span className="photos-count">{count} photos ready</span>
                <span className="photos-size-diff">
                  {formatBytes(before)} → {formatBytes(after)}
                </span>
              </div>
            )}

            {photos.length > 0 && (
              <div className="photo-grid">
                {photos.map((p) => (
                  <div key={p.id} className="photo-item">
                    <img src={p.thumbUrl} alt="" className="photo-thumb" />
                    <button
                      type="button"
                      className="photo-remove-btn"
                      onClick={(e) => {
                        e.stopPropagation()
                        remove(p)
                      }}
                      title="Remove photo"
                    >
                      <img src={iconXSvg} alt="" width={12} height={12} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {progress && (
              <div className="upload-progress-info">
                <span>
                  Uploading {progress.viewsDone}/{progress.viewsTotal} photos ({formatBytes(progress.bytesDone)}/
                  {formatBytes(progress.bytesTotal)})
                </span>
              </div>
            )}
          </div>

          {/* Right column: Tips & What happens next */}
          <div className="new-scene-sidebar">
            <div className="tips-card">
              <div className="tips-header">
                <img src={iconEyeSvg} alt="" width={18} height={18} />
                <h3 className="tips-title">Tips for a good scene</h3>
              </div>
              <ul className="tips-list">
                <li className="tips-item">
                  <img src={iconCheckSvg} alt="" width={16} height={16} />
                  <span>Keep everything still while you shoot.</span>
                </li>
                <li className="tips-item">
                  <img src={iconCheckSvg} alt="" width={16} height={16} />
                  <span>Overlap each photo with the last by about half.</span>
                </li>
                <li className="tips-item">
                  <img src={iconCheckSvg} alt="" width={16} height={16} />
                  <span>Shoot from a few heights, not just eye level.</span>
                </li>
                <li className="tips-item">
                  <img src={iconCheckSvg} alt="" width={16} height={16} />
                  <span>Skip mirrors and big windows when you can.</span>
                </li>
              </ul>
            </div>

            <div className="next-steps-card">
              <h3 className="next-steps-title">What happens next</h3>
              <p className="next-steps-desc">
                We build the 3D scene, find each object and name it. It usually takes 5–15 minutes,
                and you can leave this page.
              </p>
              <div className="next-steps-actions">
                <button
                  type="submit"
                  className="btn btn-primary btn-full"
                  disabled={count === 0 || countError !== null || preparing !== null || busy}
                >
                  {busy ? 'Starting…' : 'Build 3D scene'}
                </button>
                <button
                  type="button"
                  className="btn btn-ghost btn-full"
                  onClick={onCancel}
                  disabled={busy}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </form>
      </div>
    </div>
  )
}
