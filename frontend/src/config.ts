import type { BackendConfig } from './api'

/** The backend URL and token, fixed at build time from `frontend/.env`. */
export const backendConfig: BackendConfig = {
  url: (import.meta.env.VITE_BACKEND_URL ?? '').trim(),
  token: (import.meta.env.VITE_BACKEND_TOKEN ?? '').trim(),
}
