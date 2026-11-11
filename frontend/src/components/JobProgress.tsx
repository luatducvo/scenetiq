import { useEffect, useState } from 'react'
import { cancelJob, getJob, type BackendConfig, type Job } from '../api'
import { formatDuration, secondsBetween } from '../duration'
import iconArrowLeftSvg from '../assets/icons/icon-arrow-left.svg'
import iconCheckSvg from '../assets/icons/icon-check.svg'
import iconLoaderSvg from '../assets/icons/icon-loader.svg'
import iconEyeSvg from '../assets/icons/icon-eye.svg'
import iconTagSvg from '../assets/icons/icon-tag.svg'
import iconCloudSvg from '../assets/icons/icon-cloud.svg'

interface Props {
  config: BackendConfig
  jobId: string
  onSucceeded: (sceneId: string) => void
  onBack: () => void
}

const TERMINAL = new Set(['succeeded', 'failed', 'cancelled'])

interface PipelineStep {
  id: string
  title: string
  desc: string
  backendStages: string[]
  icon: string
}

const PIPELINE_STEPS: PipelineStep[] = [
  {
    id: 'building-3d',
    title: 'Building 3D',
    desc: 'Working out where each photo was taken',
    backendStages: ['reconstructing', 'fusing'],
    icon: iconCheckSvg,
  },
  {
    id: 'leveling-room',
    title: 'Leveling the room',
    desc: 'Finding which way is up',
    backendStages: ['aligning'],
    icon: iconCheckSvg,
  },
  {
    id: 'finding-objects',
    title: 'Finding objects',
    desc: 'Spotting each separate thing in the photos',
    backendStages: ['superpoints', 'masking', 'merging'],
    icon: iconLoaderSvg,
  },
  {
    id: 'learning-looks',
    title: 'Learning how things look',
    desc: 'So you can search by colour, shape or use',
    backendStages: ['embedding'],
    icon: iconEyeSvg,
  },
  {
    id: 'naming-objects',
    title: 'Naming objects',
    desc: 'Writing a short name for each one',
    backendStages: ['captioning'],
    icon: iconTagSvg,
  },
]

function useTick(active: boolean): void {
  const [, setTick] = useState(0)
  useEffect(() => {
    if (!active) return
    const timer = window.setInterval(() => setTick((t) => t + 1), 1000)
    return () => window.clearInterval(timer)
  }, [active])
}

export function JobProgress({ config, jobId, onSucceeded, onBack }: Props) {
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

  const isStopping = running && (stopping || job.cancel_requested)

  async function handleCancel() {
    if (!window.confirm('Cancel this job? Its photos and unfinished reconstruction will be deleted.')) return
    setCancelError(null)
    try {
      await cancelJob(config, jobId)
      setStopping(true)
    } catch (err) {
      setCancelError(String(err))
    }
  }

  // Determine current active pipeline step
  const currentBackendStage = job?.stage ?? ''
  let activeStepIndex = 0
  for (let i = 0; i < PIPELINE_STEPS.length; i++) {
    if (PIPELINE_STEPS[i].backendStages.includes(currentBackendStage)) {
      activeStepIndex = i
      break
    }
  }
  if (job?.status === 'succeeded') {
    activeStepIndex = PIPELINE_STEPS.length
  }

  const progressPercent = Math.min(
    100,
    Math.max(5, Math.round(((activeStepIndex + 0.5) / PIPELINE_STEPS.length) * 100)),
  )

  const startedAgo = job?.started_at
    ? formatDuration(secondsBetween(job.started_at, new Date().toISOString()))
    : ''

  const timings = new Map(job?.stages.map((s) => [s.stage, s]) ?? [])

  return (
    <div className="processing-page">
      <div className="processing-container">
        <button className="btn btn-ghost back-btn" onClick={onBack} type="button">
          <img src={iconArrowLeftSvg} alt="" width={18} height={18} />
          <span>Scenes</span>
        </button>

        {pollError && <div className="banner error-banner">{pollError}</div>}
        {cancelError && <div className="banner error-banner">{cancelError}</div>}

        <div className="processing-card">
          <div className="processing-card-header">
            <h1 className="processing-scene-title">
              {job?.status === 'failed' ? 'Build failed' : 'Building scene'}
            </h1>
            <p className="processing-scene-meta">
              {startedAgo ? `Started ${startedAgo} ago` : 'Waiting for worker…'}
              {isStopping && ' · stopping…'}
            </p>
          </div>

          <div className="processing-progress-section">
            <div className="progress-label-row">
              <span className="progress-step-text">
                Step {Math.min(activeStepIndex + 1, PIPELINE_STEPS.length)} of {PIPELINE_STEPS.length}
              </span>
              <span className="progress-estimate-text">
                {job?.status === 'succeeded'
                  ? 'Complete'
                  : job?.status === 'failed'
                    ? 'Stopped'
                    : 'In progress'}
              </span>
            </div>
            <div className="progress-bar-track">
              <div
                className="progress-bar-fill"
                style={{ width: job?.status === 'succeeded' ? '100%' : `${progressPercent}%` }}
              />
            </div>
          </div>

          {/* Steps List matching Figma */}
          <div className="processing-steps-list">
            {PIPELINE_STEPS.map((step, idx) => {
              const isCompleted = idx < activeStepIndex
              const isCurrent = idx === activeStepIndex && running
              const isPending = idx > activeStepIndex

              // Find step duration from backend stages
              let stepDuration = ''
              for (const stageName of step.backendStages) {
                const s = timings.get(stageName)
                if (s?.started_at) {
                  stepDuration = formatDuration(secondsBetween(s.started_at, s.finished_at))
                }
              }

              return (
                <div key={step.id} className="processing-step-row">
                  <div
                    className={`step-icon-wrap ${
                      isCompleted ? 'step-completed' : isCurrent ? 'step-current' : 'step-pending'
                    }`}
                  >
                    {isCompleted && (
                      <img src={iconCheckSvg} alt="" width={16} height={16} />
                    )}
                    {isCurrent && (
                      <img src={iconLoaderSvg} alt="" className="spin" width={16} height={16} />
                    )}
                    {isPending && (
                      <img src={step.icon} alt="" width={16} height={16} style={{ opacity: 0.5 }} />
                    )}
                  </div>

                  <div className="step-info">
                    <span className={`step-title ${isCurrent ? 'step-title-active' : ''}`}>
                      {step.title}
                    </span>
                    <span className="step-desc">{step.desc}</span>
                  </div>

                  <div className="step-time">
                    {isCompleted && stepDuration && (
                      <span className="step-time-completed">{stepDuration}</span>
                    )}
                    {isCurrent && (
                      <span className="step-time-active">
                        {stepDuration ? `${stepDuration} so far` : 'working…'}
                      </span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>

          <div className="cloud-note-banner">
            <img src={iconCloudSvg} alt="" width={18} height={18} />
            <p className="cloud-note-text">
              You can close this page. We'll keep working, and the scene shows up in Your scenes
              when it's ready.
            </p>
          </div>

          {running && (
            <div className="processing-card-actions">
              <button
                type="button"
                className="btn btn-secondary danger"
                onClick={handleCancel}
                disabled={isStopping}
              >
                {isStopping ? 'Stopping…' : 'Cancel build'}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
