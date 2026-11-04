/** Typed client for the SceneTIQ backend. */

export interface BackendConfig {
  url: string
  token: string
}

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed'

export interface Job {
  job_id: string
  scene_id: string
  status: JobStatus
  stage: string | null
  error: string | null
}

export interface SceneSummary {
  scene_id: string
  job_id: string
  name: string
  created_at: string
  status: JobStatus
}

export const MIN_VIEWS = 10
export const MAX_VIEWS = 300

/** Raised for any non-2xx response, carrying the backend's reason. */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request(cfg: BackendConfig, path: string, init: RequestInit = {}): Promise<Response> {
  const base = cfg.url.replace(/\/+$/, '')
  const resp = await fetch(`${base}${path}`, {
    ...init,
    headers: { ...(init.headers ?? {}), Authorization: `Bearer ${cfg.token}` },
  })
  if (!resp.ok) {
    let detail = resp.statusText
    try {
      const body = await resp.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      /* not JSON */
    }
    throw new ApiError(resp.status, detail)
  }
  return resp
}

/** How many Views are sent at once. */
const VIEW_CONCURRENCY = 8
const VIEW_ATTEMPTS = 5
const RETRY_BASE_MS = 500

export interface UploadProgress {
  viewsDone: number
  viewsTotal: number
  bytesDone: number
  bytesTotal: number
  /** Average speed since the first View started, in bytes per second. */
  bytesPerSecond: number
}

/** Network failures, rate limits and gateway errors can pass; validation errors cannot. */
function isRetryable(err: unknown): boolean {
  if (err instanceof DOMException && err.name === 'AbortError') return false
  if (err instanceof ApiError) return err.status === 408 || err.status === 429 || err.status >= 500
  return true
}

function wait(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms)
    signal.addEventListener('abort', () => {
      clearTimeout(timer)
      reject(signal.reason)
    })
  })
}

async function sendView(
  cfg: BackendConfig,
  captureId: string,
  index: number,
  image: File,
  signal: AbortSignal,
): Promise<void> {
  const form = new FormData()
  form.append('index', String(index))
  form.append('image', image, image.name)
  for (let attempt = 1; ; attempt++) {
    try {
      await request(cfg, `/captures/${captureId}/views`, { method: 'POST', body: form, signal })
      return
    } catch (err) {
      if (attempt >= VIEW_ATTEMPTS || !isRetryable(err)) throw err
      const backoff = RETRY_BASE_MS * 2 ** (attempt - 1)
      await wait(backoff + Math.random() * backoff, signal)
    }
  }
}

/**
 * Send a whole Capture, one request per View, and start its Scene's job.
 *
 * Views go separately because Cloudflare refuses large request bodies (see
 * docs/adr/0006). The first View that fails for good cancels the rest.
 */
export async function createScene(
  cfg: BackendConfig,
  name: string,
  images: File[],
  onProgress?: (progress: UploadProgress) => void,
): Promise<{ scene_id: string; job_id: string }> {
  const { capture_id: captureId } = (await (
    await request(cfg, '/captures', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ view_count: images.length }),
    })
  ).json()) as { capture_id: string }

  const bytesTotal = images.reduce((sum, img) => sum + img.size, 0)
  const startedAt = performance.now()
  const controller = new AbortController()
  let next = 0
  let viewsDone = 0
  let bytesDone = 0

  async function worker(): Promise<void> {
    while (next < images.length && !controller.signal.aborted) {
      const index = next++
      const image = images[index]
      await sendView(cfg, captureId, index, image, controller.signal)
      viewsDone += 1
      bytesDone += image.size
      const seconds = (performance.now() - startedAt) / 1000
      onProgress?.({
        viewsDone,
        viewsTotal: images.length,
        bytesDone,
        bytesTotal,
        bytesPerSecond: seconds > 0 ? bytesDone / seconds : 0,
      })
    }
  }

  const workers = Array.from({ length: Math.min(VIEW_CONCURRENCY, images.length) }, () => worker())
  try {
    await Promise.all(workers)
  } catch (err) {
    controller.abort()
    await Promise.allSettled(workers)
    throw err
  }

  return (
    await request(cfg, '/scenes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, capture_id: captureId }),
    })
  ).json()
}

export async function getJob(cfg: BackendConfig, jobId: string): Promise<Job> {
  return (await request(cfg, `/jobs/${jobId}`)).json()
}

export async function listScenes(cfg: BackendConfig): Promise<SceneSummary[]> {
  return (await request(cfg, '/scenes')).json()
}

/** Fetch a Scene's point cloud as raw binary PLY. */
export async function getPointCloud(cfg: BackendConfig, sceneId: string): Promise<ArrayBuffer> {
  return (await request(cfg, `/scenes/${sceneId}/points`)).arrayBuffer()
}

export async function deleteScene(cfg: BackendConfig, sceneId: string): Promise<void> {
  await request(cfg, `/scenes/${sceneId}`, { method: 'DELETE' })
}

export interface InstanceInfo {
  instance_id: number
  centroid: [number, number, number]
  bbox_min: [number, number, number]
  bbox_max: [number, number, number]
}

export interface MatchInfo {
  instance_id: number
  score: number
}

export interface QueryResult {
  matches: MatchInfo[]
  low_confidence: boolean
}

/** Per-point Instance ids (-1 = none), in the same order as the point cloud. */
export async function getInstanceIds(cfg: BackendConfig, sceneId: string): Promise<Int32Array> {
  const buf = await (await request(cfg, `/scenes/${sceneId}/instance-ids`)).arrayBuffer()
  return new Int32Array(buf)
}

export async function getInstances(cfg: BackendConfig, sceneId: string): Promise<InstanceInfo[]> {
  return (await request(cfg, `/scenes/${sceneId}/instances`)).json()
}

export async function queryScene(
  cfg: BackendConfig,
  sceneId: string,
  text: string,
): Promise<QueryResult> {
  return (
    await request(cfg, `/scenes/${sceneId}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    })
  ).json()
}
