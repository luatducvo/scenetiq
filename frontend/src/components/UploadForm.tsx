import { useEffect, useRef, useState } from 'react'
import {
  ApiError,
  createScene,
  getOptions,
  type BackendConfig,
  type BackendOptions,
  type UploadProgress,
} from '../api'
import { formatBytes, preparePhotos, releasePhoto, type PreparedPhoto } from '../photos'

interface Props {
  config: BackendConfig
  onStarted: (jobId: string, sceneId: string) => void
}

const ACCEPTED = /\.(jpe?g|png)$/i

/**
 * Pick the photos of a Capture, preview them as they will be sent (oriented, shrunk and
 * stripped of metadata in the browser, ADR 0010), and upload them.
 */
export function UploadForm({ config, onStarted }: Props) {
  const [options, setOptions] = useState<BackendOptions | null>(null)
  const [optionsError, setOptionsError] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [photos, setPhotos] = useState<PreparedPhoto[]>([])
  const [rejected, setRejected] = useState<string[]>([])
  const [preparing, setPreparing] = useState<{ done: number; total: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState<UploadProgress | null>(null)
  // The latest selection; results of an older one are dropped.
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

  async function choose(list: FileList | null) {
    if (!options) return
    const files = Array.from(list ?? [])
    const run = ++selection.current
    photos.forEach(releasePhoto)
    setPhotos([])
    setError(null)
    const images = files.filter((f) => ACCEPTED.test(f.name))
    const notImages = files.filter((f) => !ACCEPTED.test(f.name)).map((f) => f.name)
    setRejected(notImages)
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
    setRejected([...notImages, ...result.failed])
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

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    if (countError || count === 0 || preparing) return
    setBusy(true)
    setError(null)
    setProgress(null)
    try {
      const files = photos.map((p) => p.file)
      const { job_id, scene_id } = await createScene(config, name.trim(), files, setProgress)
      onStarted(job_id, scene_id)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <form className="panel" onSubmit={submit}>
      <h2>New Scene</h2>
      {optionsError && <p className="error">Cannot read the backend's options: {optionsError}</p>}
      <label>
        Name
        <input value={name} onChange={(e) => setName(e.target.value)} required maxLength={200} />
      </label>
      <label>
        Photos of one static Scene (JPG/PNG)
        <input
          type="file"
          multiple
          accept=".jpg,.jpeg,.png"
          disabled={!options || busy}
          onChange={(e) => choose(e.target.files)}
        />
      </label>
      {preparing && (
        <p className="muted">
          <progress value={preparing.done} max={preparing.total} /> Preparing {preparing.done}/
          {preparing.total} photos…
        </p>
      )}
      {rejected.length > 0 && (
        <p className="error">Left out, not a readable JPG or PNG: {rejected.join(', ')}</p>
      )}
      {count > 0 && (
        <>
          <p className="muted">
            {count} photos · {formatBytes(before)} → {formatBytes(after)}, shrunk to at most{' '}
            {options?.upload_max_side} px, without metadata
          </p>
          <ul className="photo-grid">
            {photos.map((p) => (
              <li key={p.id} title={`${p.file.name} · ${p.width}×${p.height}`}>
                <img src={p.thumbUrl} alt={p.file.name} loading="lazy" />
                <span className="photo-size">
                  {formatBytes(p.originalBytes)} → {formatBytes(p.file.size)}
                </span>
                <button
                  type="button"
                  className="photo-remove"
                  aria-label={`Remove ${p.file.name}`}
                  disabled={busy}
                  onClick={() => remove(p)}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
      {countError && <p className="error">{countError}</p>}
      {busy && (
        <p className="muted">
          <progress value={progress?.bytesDone ?? 0} max={progress?.bytesTotal ?? 1} />{' '}
          {progress?.viewsDone ?? 0}/{count} photos
          {progress && ` · ${(progress.bytesPerSecond / 1e6).toFixed(1)} MB/s`}
        </p>
      )}
      {error && <p className="error">{error}</p>}
      <button
        type="submit"
        disabled={busy || !options || preparing !== null || count === 0 || countError !== null}
      >
        {busy ? 'Uploading…' : 'Upload and reconstruct'}
      </button>
    </form>
  )
}
