import { useEffect, useState } from 'react'
import { getJob, type BackendConfig, type Job } from '../api'

const STAGES = ['reconstructing', 'fusing', 'superpoints', 'masking', 'merging', 'embedding']

interface Props {
  config: BackendConfig
  jobId: string
  onSucceeded: (sceneId: string) => void
}

/** Poll a job and show which pipeline stage it is in. */
export function JobProgress({ config, jobId, onSucceeded }: Props) {
  const [job, setJob] = useState<Job | null>(null)
  const [pollError, setPollError] = useState<string | null>(null)

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
        if (next.status === 'failed') return
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

  const current = job?.stage ? STAGES.indexOf(job.stage) : -1

  return (
    <div className="panel">
      <h2>Processing</h2>
      <p>
        Status: <strong>{job?.status ?? 'connecting'}</strong>
      </p>
      <ol className="stages">
        {STAGES.map((s, i) => (
          <li
            key={s}
            className={
              job?.status === 'succeeded' || i < current ? 'done' : i === current ? 'active' : ''
            }
          >
            {s}
          </li>
        ))}
      </ol>
      {job?.status === 'failed' && <p className="error">Failed: {job.error}</p>}
      {pollError && <p className="error">Cannot reach backend: {pollError}</p>}
    </div>
  )
}
