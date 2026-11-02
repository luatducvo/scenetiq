import { useState } from 'react'
import type { BackendConfig } from '../api'

interface Props {
  config: BackendConfig
  onSave: (cfg: BackendConfig) => void
}

/** Edit the backend URL and shared token. */
export function SettingsPanel({ config, onSave }: Props) {
  const [url, setUrl] = useState(config.url)
  const [token, setToken] = useState(config.token)

  return (
    <form
      className="panel"
      onSubmit={(e) => {
        e.preventDefault()
        onSave({ url: url.trim(), token: token.trim() })
      }}
    >
      <h2>Backend</h2>
      <label>
        URL
        <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://scenetiq.example.com" required />
      </label>
      <label>
        Token
        <input type="password" value={token} onChange={(e) => setToken(e.target.value)} required />
      </label>
      <button type="submit">Save</button>
    </form>
  )
}
