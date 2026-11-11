/** Typed client for the SceneTIQ backend. */

export interface BackendConfig {
  url: string
  token: string
}

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'

/** When one Stage of a job started and ended (null while it runs); ISO 8601 UTC. */
export interface StageTiming {
  stage: string
  started_at: string
  finished_at: string | null
}

/** One line of a job's summary, with the Stage running when it was recorded. */
export interface JobNote {
  at: string
  stage: string | null
  message: string
}

export interface Job {
  job_id: string
  scene_id: string
  status: JobStatus
  stage: string | null
  error: string | null
  started_at: string | null
  finished_at: string | null
  /** Whether this unfinished job was asked to stop and has not yet. */
  cancel_requested: boolean
  /** Every Stage that started, in order. */
  stages: StageTiming[]
  notes: JobNote[]
}

export interface SceneSummary {
  scene_id: string
  job_id: string
  name: string
  created_at: string
  status: JobStatus
  /** When its job started and ended, ISO 8601 UTC; null until then. */
  started_at: string | null
  finished_at: string | null
  /** How many Instances its Reconstruction has; null unless its job succeeded. */
  instance_count: number | null
}

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

const PING_TIMEOUT_MS = 10_000

export type BackendHealth =
  | { state: 'online'; latencyMs: number; jobRunning: boolean }
  | { state: 'unauthorized' }
  | { state: 'offline'; reason: string }

/**
 * Ping the backend; never throws.
 *
 * A stopped tunnel gets a Cloudflare error page without CORS headers, which the
 * browser reports as a network error, so that case has no HTTP status to show.
 */
export async function pingBackend(cfg: BackendConfig): Promise<BackendHealth> {
  const startedAt = performance.now()
  try {
    const resp = await request(cfg, '/health', { signal: AbortSignal.timeout(PING_TIMEOUT_MS) })
    const body = (await resp.json()) as { job_running: boolean }
    return {
      state: 'online',
      latencyMs: Math.round(performance.now() - startedAt),
      jobRunning: body.job_running,
    }
  } catch (err) {
    if (err instanceof ApiError) {
      return err.status === 401
        ? { state: 'unauthorized' }
        : { state: 'offline', reason: `HTTP ${err.status}` }
    }
    const timedOut = err instanceof DOMException && err.name === 'TimeoutError'
    return { state: 'offline', reason: timedOut ? 'no answer in time' : 'unreachable' }
  }
}

/** The settings the frontend configures itself from (GET /options). */
export interface BackendOptions {
  min_views: number
  max_views: number
  /** Photos are shrunk so that their long side is at most this many pixels (ADR 0010). */
  upload_max_side: number
  /** JPEG quality in (0, 1] that photos are re-encoded at. */
  upload_jpeg_quality: number
  /** How many Matches a Query returns when it does not ask, and at most. */
  query_top_k: number
  query_max_top_k: number
}

export async function getOptions(cfg: BackendConfig): Promise<BackendOptions> {
  return (await request(cfg, '/options')).json()
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

/** Ask a queued or running job to stop at its next checkpoint; it stops later, not now. */
export async function cancelJob(cfg: BackendConfig, jobId: string): Promise<void> {
  await request(cfg, `/jobs/${jobId}/cancel`, { method: 'POST' })
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
  /** The Name of the Instance's Caption, or null when it has none (ADR 0008). */
  name: string | null
}

/** An Instance a Match was measured against, and the phrase of the object it stands for. */
export interface AnchorInfo {
  instance_id: number
  phrase: string
}

/** One of the Instances returned for a Query, at most K (ADR 0009, ADR 0011). */
export interface MatchInfo {
  instance_id: number
  /**
   * Fused value as a fraction of the best Match's, so the best Match scores 1, or the
   * assignment score when Relations were applied; relative to this Query only.
   */
  score: number
  image_score: number
  /** Similarity of the best-matching Description, or null without a Caption. */
  caption_score: number | null
  /** The Aspect of that Description, or null without a Caption. */
  caption_aspect: Aspect | null
  /** The Anchors of the Match's assignment; empty when no Relation was applied. */
  anchors: AnchorInfo[]
}

/** A fixed side of an Instance that a Caption describes. */
export type Aspect = 'identity' | 'appearance' | 'material' | 'use'

export interface QueryResult {
  matches: MatchInfo[]
  /** Whether the Query's Relations decided the Matches. */
  relations_applied: boolean
  /**
   * Whether the parsed Query had a Relation. Stated but not applied means no set of
   * Instances could satisfy the Relations, and the Query was matched as plain text.
   */
  relations_stated: boolean
  /** The parts of the Query no Relation can express, as the parser quoted them; never applied. */
  unsupported: string[]
  /** Whether the parser could not read the Query, which was then matched as plain text. */
  parse_failed: boolean
}

/** Per-point Instance ids (-1 = none), in the same order as the point cloud. */
export async function getInstanceIds(cfg: BackendConfig, sceneId: string): Promise<Int32Array> {
  const buf = await (await request(cfg, `/scenes/${sceneId}/instance-ids`)).arrayBuffer()
  return new Int32Array(buf)
}

export async function getInstances(cfg: BackendConfig, sceneId: string): Promise<InstanceInfo[]> {
  return (await request(cfg, `/scenes/${sceneId}/instances`)).json()
}

/** Run a Query; `topK` asks for that many Matches instead of the backend's default. */
export async function queryScene(
  cfg: BackendConfig,
  sceneId: string,
  text: string,
  topK?: number,
): Promise<QueryResult> {
  return (
    await request(cfg, `/scenes/${sceneId}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(topK === undefined ? { text } : { text, top_k: topK }),
    })
  ).json()
}
