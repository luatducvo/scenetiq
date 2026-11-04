import { useState } from 'react'
import { ApiError, createScene, type BackendConfig, type UploadProgress } from '../api'

interface Props {
  config: BackendConfig
  onStarted: (jobId: string, sceneId: string) => void
}

const ACCEPTED = /\.(jpe?g|png)$/i

/** Upload a named Capture of JPG/PNG photos; the backend refuses a count outside its limits. */
export function UploadForm({ config, onStarted }: Props) {
  const [name, setName] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState<UploadProgress | null>(null)

  const badFile = files.find((f) => !ACCEPTED.test(f.name))
  const clientError = badFile ? `${badFile.name} is not a JPG or PNG file.` : null

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    if (clientError || files.length === 0) return
    setBusy(true)
    setError(null)
    setProgress(null)
    try {
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
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
        />
      </label>
      {files.length > 0 && <p className="muted">{files.length} photos selected</p>}
      {busy && (
        <p className="muted">
          <progress value={progress?.bytesDone ?? 0} max={progress?.bytesTotal ?? 1} />{' '}
          {progress?.viewsDone ?? 0}/{files.length} photos
          {progress && ` · ${(progress.bytesPerSecond / 1e6).toFixed(1)} MB/s`}
        </p>
      )}
      {(clientError ?? error) && <p className="error">{clientError ?? error}</p>}
      <button type="submit" disabled={busy || files.length === 0 || clientError !== null}>
        {busy ? 'Uploading…' : 'Upload and reconstruct'}
      </button>
    </form>
  )
}
