/**
 * Prepare photos in the browser before upload (ADR 0010): apply the EXIF orientation,
 * shrink to a maximum long side, and re-encode as JPEG, which drops every metadata field.
 */

/** A photo ready to upload, with a small preview. */
export interface PreparedPhoto {
  /** Unique within one selection, for React keys and removal. */
  id: string
  /** The prepared JPEG, named like the original with a `.jpg` extension. */
  file: File
  /** Object URL of a small JPEG preview; revoke it with {@link releasePhoto}. */
  thumbUrl: string
  originalBytes: number
  width: number
  height: number
}

export interface PrepareOptions {
  /** Longest side in pixels; a smaller photo keeps its size. */
  maxSide: number
  /** JPEG quality in (0, 1]. */
  quality: number
}

export interface PrepareResult {
  /** The prepared photos, in the order of the input files. */
  photos: PreparedPhoto[]
  /** Names of the files the browser could not decode. */
  failed: string[]
}

const THUMB_SIDE = 160
const THUMB_QUALITY = 0.7
const CONCURRENCY = 3

/** The size of a `width` x `height` image scaled to fit `maxSide`, never enlarged. */
export function fitWithin(width: number, height: number, maxSide: number): [number, number] {
  const scale = Math.min(1, maxSide / Math.max(width, height))
  return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))]
}

function encode(
  bitmap: ImageBitmap,
  width: number,
  height: number,
  quality: number,
): Promise<Blob> {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) return Promise.reject(new Error('no 2D canvas'))
  // JPEG has no alpha: a transparent PNG would otherwise turn black.
  ctx.fillStyle = '#fff'
  ctx.fillRect(0, 0, width, height)
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(bitmap, 0, 0, width, height)
  return new Promise((resolve, reject) =>
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('JPEG encoding failed'))),
      'image/jpeg',
      quality,
    ),
  )
}

function jpegName(name: string): string {
  const dot = name.lastIndexOf('.')
  return `${dot > 0 ? name.slice(0, dot) : name}.jpg`
}

/** Orient, shrink and re-encode one photo. Rejects when the browser cannot decode it. */
export async function preparePhoto(file: File, opts: PrepareOptions): Promise<PreparedPhoto> {
  // `from-image` applies the EXIF orientation to the pixels.
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' })
  try {
    const [width, height] = fitWithin(bitmap.width, bitmap.height, opts.maxSide)
    const [thumbW, thumbH] = fitWithin(bitmap.width, bitmap.height, THUMB_SIDE)
    const [full, thumb] = await Promise.all([
      encode(bitmap, width, height, opts.quality),
      encode(bitmap, thumbW, thumbH, THUMB_QUALITY),
    ])
    return {
      id: crypto.randomUUID(),
      file: new File([full], jpegName(file.name), { type: 'image/jpeg' }),
      thumbUrl: URL.createObjectURL(thumb),
      originalBytes: file.size,
      width,
      height,
    }
  } finally {
    bitmap.close()
  }
}

/**
 * Prepare every file, a few at a time, keeping their order.
 *
 * @param onProgress Called after each file with how many are done.
 */
export async function preparePhotos(
  files: File[],
  opts: PrepareOptions,
  onProgress?: (done: number, total: number) => void,
): Promise<PrepareResult> {
  const results: (PreparedPhoto | null)[] = new Array(files.length).fill(null)
  const failed: string[] = []
  let next = 0
  let done = 0

  async function worker(): Promise<void> {
    while (next < files.length) {
      const index = next++
      try {
        results[index] = await preparePhoto(files[index], opts)
      } catch {
        failed.push(files[index].name)
      }
      done += 1
      onProgress?.(done, files.length)
    }
  }

  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, files.length) }, () => worker()))
  return { photos: results.filter((p): p is PreparedPhoto => p !== null), failed }
}

/** Free a photo's preview. */
export function releasePhoto(photo: PreparedPhoto): void {
  URL.revokeObjectURL(photo.thumbUrl)
}

/** A byte count as "840 KB" or "3.2 MB". */
export function formatBytes(bytes: number): string {
  return bytes < 1e6 ? `${Math.round(bytes / 1e3)} KB` : `${(bytes / 1e6).toFixed(1)} MB`
}
