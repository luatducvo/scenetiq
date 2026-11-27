import { useEffect, useState } from 'react'
import { pingBackend, type BackendConfig, type BackendHealth } from '../api'

interface Props {
  config: BackendConfig
}

export function BackendStatus({ config }: Props) {
  const [health, setHealth] = useState<BackendHealth | null>(null)
  const [pinging, setPinging] = useState(false)

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined

    async function probe() {
      setPinging(true)
      try {
        const result = await pingBackend(config)
        if (!cancelled) setHealth(result)
      } catch {
        if (!cancelled) setHealth({ state: 'offline', reason: 'Unreachable' })
      } finally {
        if (!cancelled) setPinging(false)
      }
      timer = window.setTimeout(probe, 10000)
    }

    probe()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [config])

  const isOnline = health?.state === 'online'
  const isChecking = !health || pinging
  const statusClass = isChecking ? 'checking' : isOnline ? 'online' : 'offline'
  const label = isChecking && !health ? 'Connecting' : isOnline ? 'Connected' : 'Offline'

  return (
    <div className={`backend-status status-${statusClass}`} title={`Backend: ${config.url}`}>
      <span className={`status-dot ${statusClass}`} />
      <span className="status-label">{label}</span>
    </div>
  )
}
