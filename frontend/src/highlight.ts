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

/** The colour of every Anchor, unlike any Match colour (RGB 0-1). */
export const ANCHOR_COLOR: [number, number, number] = [0.55, 0.4, 1.0]

/** The Instance ids of the Matches' Anchors that are not Matches themselves. */
export function anchorIds(matches: MatchInfo[]): Set<number> {
  const matched = new Set(matches.map((m) => m.instance_id))
  return new Set(
    matches.flatMap((m) => m.anchors.map((a) => a.instance_id)).filter((id) => !matched.has(id)),
  )
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

/** Colour every point by its Instance, to inspect how the Scene was split into Instances. */
export function instanceColors(instanceIds: Int32Array): Float32Array {
  const out = new Float32Array(instanceIds.length * 3)
  const cache = new Map<number, [number, number, number]>()
  for (let p = 0; p < instanceIds.length; p++) {
    const id = instanceIds[p]
    let color = cache.get(id)
    if (!color) {
      color = instanceColor(id)
      cache.set(id, color)
    }
    out[3 * p] = color[0]
    out[3 * p + 1] = color[1]
    out[3 * p + 2] = color[2]
  }
  return out
}

/**
 * Colour points for display: Matched Instances get their rank's colour, their Anchors the
 * Anchor colour, and everything else is the original colour, dimmed while any Match is shown.
 */
export function highlightColors(
  base: Float32Array,
  instanceIds: Int32Array,
  matches: MatchInfo[],
): Float32Array {
  if (matches.length === 0) return base.slice()
  const rankOf = new Map(matches.map((m, i) => [m.instance_id, i]))
  const anchors = anchorIds(matches)
  const out = new Float32Array(base.length)
  for (let p = 0; p < instanceIds.length; p++) {
    const rank = rankOf.get(instanceIds[p])
    if (anchors.has(instanceIds[p])) {
      out[3 * p] = ANCHOR_COLOR[0]
      out[3 * p + 1] = ANCHOR_COLOR[1]
      out[3 * p + 2] = ANCHOR_COLOR[2]
    } else if (rank === undefined) {
      out[3 * p] = base[3 * p] * 0.35
      out[3 * p + 1] = base[3 * p + 1] * 0.35
      out[3 * p + 2] = base[3 * p + 2] * 0.35
    } else {
      const [r, g, b] = matchColor(rank)
      out[3 * p] = r
      out[3 * p + 1] = g
      out[3 * p + 2] = b
    }
  }
  return out
}
