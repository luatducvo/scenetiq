import type { MatchInfo } from './api'

/** Distinct highlight colours for ranked Matches (RGB 0-1). */
export const MATCH_COLORS: [number, number, number][] = [
  [1.0, 0.84, 0.0],
  [0.0, 0.9, 1.0],
  [1.0, 0.3, 0.9],
  [0.4, 1.0, 0.3],
  [1.0, 0.5, 0.1],
]

export function matchColor(rank: number): [number, number, number] {
  return MATCH_COLORS[rank % MATCH_COLORS.length]
}

/**
 * Colour points for display: Matched Instances get their rank's colour,
 * everything else is the original colour, dimmed while any Match is shown.
 */
export function highlightColors(
  base: Float32Array,
  instanceIds: Int32Array,
  matches: MatchInfo[],
): Float32Array {
  if (matches.length === 0) return base.slice()
  const rankOf = new Map(matches.map((m, i) => [m.instance_id, i]))
  const out = new Float32Array(base.length)
  for (let p = 0; p < instanceIds.length; p++) {
    const rank = rankOf.get(instanceIds[p])
    if (rank === undefined) {
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
