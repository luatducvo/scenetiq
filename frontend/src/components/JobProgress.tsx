import { useEffect, useState } from 'react'
import { cancelJob, getJob, type BackendConfig, type Job } from '../api'
import { formatDuration, secondsBetween } from '../duration'

const STAGES = [
  'reconstructing',
  'fusing',
  'aligning',
  'superpoints',
  'masking',
  'merging',
  'embedding',
  'captioning',
]

const TERMINAL = new Set(['succeeded', 'failed', 'cancelled'])

interface Props {
  config: BackendConfig
  jobId: string
  onSucceeded: (sceneId: string) => void
}

/** Re-render every second while `active`, so that running durations count up. */
function useTick(active: boolean): void {
  const [, setTick] = useState(0)
  useEffect(() => {
    if (!active) return
    const timer = window.setInterval(() => setTick((t) => t + 1), 1000)
    return () => window.clearInterval(timer)
  }, [active])
}

/** Poll a job and show its stages with their durations, and its notes. */
export function JobProgress({ config, jobId, onSucceeded }: Props) {
  const [job, setJob] = useState<Job | null>(null)
  const [pollError, setPollError] = useState<string | null>(null)
  const [stopping, setStopping] = useState(false)
  const [cancelError, setCancelError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined
    async function poll() {
      try {
        const next = await getJob(config, jobId)
        if (cancelled) return
        setJob(next)
        setPollError(null)
        if (next.status === 'succeeded') {
          onSucceeded(next.scene_id)
          return
        }
        if (TERMINAL.has(next.status)) return
      } catch (err) {
        if (!cancelled) setPollError(String(err))
      }
      timer = window.setTimeout(poll, 1500)
    }
    poll()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [config, jobId, onSucceeded])

  const running = job !== null && !TERMINAL.has(job.status)
  useTick(running)
  // The backend remembers a cancel request, so "stopping" survives a reload or a cancel
  // sent from the Scene list.
  const isStopping = running && (stopping || job.cancel_requested)
  // A job interrupted by a restart never got an end time: show no duration that ends now.
  const timed = running || job?.finished_at != null

  async function cancel() {
    if (!window.confirm('Cancel this job? Its photos and unfinished Reconstruction will be deleted.')) return
    setCancelError(null)
    try {
      await cancelJob(config, jobId)
      setStopping(true)
    } catch (err) {
      setCancelError(String(err))
    }
  }

  const timings = new Map(job?.stages.map((s) => [s.stage, s]) ?? [])
  const current = job?.stage ? STAGES.indexOf(job.stage) : -1

  return (
    <div className="panel">
      <h2>Processing</h2>
      <p>
        Status: <strong>{isStopping ? 'stopping' : (job?.status ?? 'connecting')}</strong>
        {job?.started_at && timed && (
          <span className="muted">
            {' · '}
            {formatDuration(secondsBetween(job.started_at, job.finished_at))}
          </span>
        )}
      </p>
      <ol className="stages">
        {STAGES.map((s, i) => {
          const timing = timings.get(s)
          const state =
            job?.status === 'succeeded' || i < current ? 'done' : i === current ? 'active' : ''
          return (
            <li key={s} className={state}>
              {s}
              {timing && (running || timing.finished_at) && (
                <span className="stage-time">
                  {' '}
                  {formatDuration(secondsBetween(timing.started_at, timing.finished_at))}
                </span>
              )}
            </li>
          )
        })}
      </ol>
      {job && job.notes.length > 0 && (
        <ul className="job-notes">
          {job.notes.map((n, i) => (
            <li key={i}>
              {n.stage && <span className="muted">{n.stage}: </span>}
              {n.message}
            </li>
          ))}
        </ul>
      )}
      {running && !isStopping && (
        <button type="button" className="danger" onClick={cancel}>
          Cancel job
        </button>
      )}
      {isStopping && (
        <p className="muted">
          Stopping at the next checkpoint: the end of the current stage, View or Instance. A
          single model call cannot be interrupted.
        </p>
      )}
      {job?.status === 'cancelled' && (
        <p className="muted">Cancelled. Its photos and unfinished Reconstruction were deleted.</p>
      )}
      {cancelError && <p className="error">Cannot cancel: {cancelError}</p>}
      {job?.status === 'failed' && <p className="error">Failed: {job.error}</p>}
      {pollError && <p className="error">Cannot reach backend: {pollError}</p>}
    </div>
  )
}
