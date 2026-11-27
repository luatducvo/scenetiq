/** Seconds between two ISO 8601 timestamps; `end` defaults to now. */
export function secondsBetween(start: string, end?: string | null): number {
  const to = end ? Date.parse(end) : Date.now()
  return Math.max(0, (to - Date.parse(start)) / 1000)
}

/** A duration as "42s", "3m 05s" or "1h 02m". */
export function formatDuration(seconds: number): string {
  const s = Math.floor(seconds)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, '0')}s`
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`
}
