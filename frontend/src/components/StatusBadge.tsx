import statusReadySvg from '../assets/icons/status-ready.svg'
import statusProcessingSvg from '../assets/icons/status-processing.svg'
import statusQueuedSvg from '../assets/icons/status-queued.svg'
import statusFailedSvg from '../assets/icons/status-failed.svg'

export type StatusType = 'ready' | 'processing' | 'queued' | 'failed' | 'succeeded' | 'running' | 'cancelled'

interface StatusBadgeProps {
  status: StatusType | string
  className?: string
}

export function StatusBadge({ status, className = '' }: StatusBadgeProps) {
  const norm = status.toLowerCase()
  let kind: 'ready' | 'processing' | 'queued' | 'failed' = 'ready'
  let label = 'Ready'
  let icon = statusReadySvg

  if (norm === 'succeeded' || norm === 'ready') {
    kind = 'ready'
    label = 'Ready'
    icon = statusReadySvg
  } else if (norm === 'running' || norm === 'processing') {
    kind = 'processing'
    label = 'Processing'
    icon = statusProcessingSvg
  } else if (norm === 'queued') {
    kind = 'queued'
    label = 'Queued'
    icon = statusQueuedSvg
  } else if (norm === 'failed' || norm === 'cancelled') {
    kind = 'failed'
    label = norm === 'cancelled' ? 'Cancelled' : 'Failed'
    icon = statusFailedSvg
  }

  return (
    <div className={`status-badge status-badge-${kind} ${className}`}>
      <img src={icon} alt="" className={`badge-icon ${kind === 'processing' ? 'spin' : ''}`} width={14} height={14} />
      <span className="badge-text">{label}</span>
    </div>
  )
}
