import * as THREE from 'three'
import type { MatchInfo } from './api'

/** Hue of the coolest heatmap colour (blue); the hottest is 0 (red). */
const COOLEST_HUE = 0.66

/**
 * The heatmap colour at `position` in [0, 1], from blue (0) through green and yellow to red
 * (1). Linear like {@link instanceColor}, because the renderer converts vertex colours to sRGB.
 */
export function heatmapColor(position: number): [number, number, number] {
  const t = Math.min(1, Math.max(0, position))
  const color = new THREE.Color().setHSL((1 - t) * COOLEST_HUE, 1, 0.5, THREE.SRGBColorSpace)
  return [color.r, color.g, color.b]
}

/**
 * The heatmap as a CSS gradient, best (red) on the left to lowest (blue) on the right, for
 * the colour key. Same scale as {@link heatmapColor}, but in sRGB, as CSS expects.
 */
export function heatmapGradient(): string {
  const stops = [0, 0.25, 0.5, 0.75, 1].map((s) =>
    new THREE.Color().setHSL(s * COOLEST_HUE, 1, 0.5, THREE.SRGBColorSpace).getStyle(THREE.SRGBColorSpace),
  )
  return `linear-gradient(to right, ${stops.join(', ')})`
}

/**
 * Each Match's heatmap position, stretched over the scores shown: the best Match is 1 and
 * the lowest 0. Scores of one Query are often close together, so a fixed 0-1 scale would
 * give them all one colour. When every score is equal, including a single Match, all are 1.
 */
export function heatmapPositions(matches: MatchInfo[]): number[] {
  const scores = matches.map((m) => m.score)
  const lo = Math.min(...scores)
  const hi = Math.max(...scores)
  return scores.map((s) => (hi - lo > 1e-9 ? (s - lo) / (hi - lo) : 1))
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

/**
 * Colour points for display: Matched Instances get their heatmap colour (see
 * {@link heatmapPositions}); every other point keeps its original colour.
 */
export function highlightColors(
  base: Float32Array,
  instanceIds: Int32Array,
  matches: MatchInfo[],
): Float32Array {
  if (matches.length === 0) return base.slice()
  const positions = heatmapPositions(matches)
  const colorOf = new Map(matches.map((m, i) => [m.instance_id, heatmapColor(positions[i])]))
  const out = base.slice()
  for (let p = 0; p < instanceIds.length; p++) {
    const color = colorOf.get(instanceIds[p])
    if (color === undefined) continue
    const [r, g, b] = color
    out[3 * p] = r
    out[3 * p + 1] = g
    out[3 * p + 2] = b
  }
  return out
}
