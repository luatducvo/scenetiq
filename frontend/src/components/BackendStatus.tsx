import { useEffect, useState } from 'react'
import { pingBackend, type BackendConfig, type BackendHealth } from '../api'

const PING_INTERVAL_MS = 15_000

function describe(health: BackendHealth | null): { tone: string; text: string } {
  if (health === null) return { tone: 'unknown', text: 'Checking backend…' }
  switch (health.state) {
    case 'online':
      return {
        tone: 'online',
        text: `Online · ${health.latencyMs} ms${health.jobRunning ? ' · job running' : ''}`,
      }
    case 'unauthorized':
      return { tone: 'offline', text: 'Invalid token' }
    case 'offline':
      return { tone: 'offline', text: `Offline · ${health.reason}` }
  }
}

/** Ping the backend periodically and on click, and show whether it is healthy. */
export function BackendStatus({ config }: { config: BackendConfig }) {
  const [health, setHealth] = useState<BackendHealth | null>(null)
  const [manualPings, setManualPings] = useState(0)

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined
    async function ping() {
      const next = await pingBackend(config)
      if (cancelled) return
      setHealth(next)
      timer = window.setTimeout(ping, PING_INTERVAL_MS)
    }
    ping()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [config, manualPings])

  const { tone, text } = describe(health)
  return (
    <button
      type="button"
      className="backend-status"
      title="Ping the backend now"
      onClick={() => setManualPings((n) => n + 1)}
    >
      <span className={`status-dot ${tone}`} aria-hidden="true" />
      {text}
    </button>
  )
}
