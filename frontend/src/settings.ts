import type { BackendConfig } from './api'

const KEY = 'scenetiq.backend'

/** Load the saved backend URL and token; storage may be unavailable. */
export function loadConfig(): BackendConfig {
  try {
    const raw = localStorage.getItem(KEY)
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<BackendConfig>
      return { url: parsed.url ?? '', token: parsed.token ?? '' }
    }
  } catch {
    /* ignore */
  }
  return { url: 'http://localhost:8001', token: '' }
}

export function saveConfig(cfg: BackendConfig): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(cfg))
  } catch {
    /* ignore */
  }
}
