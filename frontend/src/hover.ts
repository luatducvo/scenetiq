import type { InstanceInfo } from './api'

/**
 * The Name to show for the point under the pointer, or null when there is none to show.
 *
 * A point outside any Instance, an Instance without a Caption, and an Instance outside
 * `shown` (when given) all give null. Pass the Matched Instance ids as `shown` while a
 * Query result is displayed, and null in the Names view, where every Instance
 * may be named.
 */
export function nameAtPoint(
  pointIndex: number,
  instanceIds: Int32Array,
  instances: Map<number, InstanceInfo>,
  shown: Set<number> | null,
): string | null {
  const instanceId = instanceIds[pointIndex]
  if (instanceId === undefined || instanceId < 0) return null
  if (shown && !shown.has(instanceId)) return null
  return instances.get(instanceId)?.name ?? null
}
