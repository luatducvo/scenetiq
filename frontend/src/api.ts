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

export async function createScene(
  cfg: BackendConfig,
  name: string,
  images: File[],
): Promise<{ scene_id: string; job_id: string }> {
  const form = new FormData()
  form.append('name', name)
  for (const img of images) form.append('images', img, img.name)
  return (await request(cfg, '/scenes', { method: 'POST', body: form })).json()
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
