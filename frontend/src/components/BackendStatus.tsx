import { useEffect, useState } from 'react'
import { pingBackend, type BackendConfig, type BackendHealth } from '../api'

interface PingResult {
  health: BackendHealth
  checkedAt: string
}

function describe(health: BackendHealth): { tone: string; text: string } {
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

/** Show the result of the latest ping: one when the page loads, then only on request. */
export function BackendStatus({ config }: { config: BackendConfig }) {
  const [result, setResult] = useState<PingResult | null>(null)
  const [pinging, setPinging] = useState(true)
  const [manualPings, setManualPings] = useState(0)

  useEffect(() => {
    let cancelled = false
    pingBackend(config).then((health) => {
      if (cancelled) return
      setResult({ health, checkedAt: new Date().toLocaleTimeString() })
      setPinging(false)
    })
    return () => {
      cancelled = true
    }
  }, [config, manualPings])

  function pingNow() {
    setPinging(true)
    setManualPings((n) => n + 1)
  }

  const { tone, text } = result
    ? describe(result.health)
    : { tone: 'unknown', text: 'Checking backend…' }
  return (
    <div className="backend-status">
      <span className={`status-dot ${tone}`} aria-hidden="true" />
      <span role="status">
        {text}
        {result && ` · checked ${result.checkedAt}`}
      </span>
      <button type="button" onClick={pingNow} disabled={pinging}>
        {pinging ? 'Pinging…' : 'Ping'}
      </button>
    </div>
  )
}
