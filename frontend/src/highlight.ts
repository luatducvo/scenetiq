import * as THREE from 'three'
import type { MatchInfo } from './api'

/** Distinct highlight colours for ranked Matches (RGB 0-1). */
const MATCH_COLORS:[number, number, number][] = [
  [1.0, 0.84, 0.0],
  [0.0, 0.9, 1.0],
  [1.0, 0.3, 0.9],
  [0.4, 1.0, 0.3],
  [1.0, 0.5, 0.1],
]

export function matchColor(rank: number): [number, number, number] {
  return MATCH_COLORS[rank % MATCH_COLORS.length]
}

/** Stepping the hue by the golden ratio keeps consecutive Instance ids far apart. */
const HUE_STEP = 0.618034
const INSTANCE_LIGHTNESS = [0.5, 0.65, 0.38]
const INSTANCE_SATURATION = 0.85
const NO_INSTANCE_LIGHTNESS = 0.3

/**
 * The colour of an Instance, which depends only on its id; a negative id means no Instance.
 *
 * The result is linear, like the colours PLYLoader returns, because the renderer converts
 * every vertex colour to sRGB. Picking in sRGB here makes what is shown match what is picked.
 */
export function instanceColor(instanceId: number): [number, number, number] {
  const color = new THREE.Color()
  if (instanceId < 0) {
    color.setHSL(0, 0, NO_INSTANCE_LIGHTNESS, THREE.SRGBColorSpace)
  } else {
    const hue = (instanceId * HUE_STEP) % 1
    const lightness = INSTANCE_LIGHTNESS[instanceId % INSTANCE_LIGHTNESS.length]
    color.setHSL(hue, INSTANCE_SATURATION, lightness, THREE.SRGBColorSpace)
  }
  return [color.r, color.g, color.b]
}

/** The colour of an Instance without a Caption in the Names view. */
const UNNAMED_COLOR: [number, number, number] = [0, 0, 0]

/**
 * Colour every point for the Names view: a named Instance by its Instance colour, an
 * Instance without a Caption black, and a point of no Instance grey.
 */
export function namesColors(instanceIds: Int32Array, named: Set<number>): Float32Array {
  const out = new Float32Array(instanceIds.length * 3)
  const cache = new Map<number, [number, number, number]>()
  for (let p = 0; p < instanceIds.length; p++) {
    const id = instanceIds[p]
    let color = cache.get(id)
    if (!color) {
      color = id >= 0 && !named.has(id) ? UNNAMED_COLOR : instanceColor(id)
      cache.set(id, color)
    }
    out[3 * p] = color[0]
    out[3 * p + 1] = color[1]
    out[3 * p + 2] = color[2]
  }
  return out
}

/** Colour points for display: Matched Instances get their rank's colour; every other point keeps its original colour. */
export function highlightColors(
  base: Float32Array,
  instanceIds: Int32Array,
  matches: MatchInfo[],
): Float32Array {
  if (matches.length === 0) return base.slice()
  const rankOf = new Map(matches.map((m, i) => [m.instance_id, i]))
  const out = base.slice()
  for (let p = 0; p < instanceIds.length; p++) {
    const rank = rankOf.get(instanceIds[p])
    if (rank === undefined) continue
    const [r, g, b] = matchColor(rank)
    out[3 * p] = r
    out[3 * p + 1] = g
    out[3 * p + 2] = b
  }
  return out
}
