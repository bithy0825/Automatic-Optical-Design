/** 点列统计(纯函数,mm 单位;显示用 fmtUm 转 μm)。 */

export function centroid(pts: ArrayLike<number>): [number, number] {
  let sx = 0, sy = 0, n = 0;
  for (let i = 0; i < pts.length; i += 2) {
    const x = pts[i], y = pts[i + 1];
    if (Number.isFinite(x) && Number.isFinite(y)) { sx += x; sy += y; n++; }
  }
  return n ? [sx / n, sy / n] : [NaN, NaN];
}

export function rmsRadius(pts: ArrayLike<number>, ref: [number, number]): number {
  let acc = 0, n = 0;
  for (let i = 0; i < pts.length; i += 2) {
    const dx = pts[i] - ref[0], dy = pts[i + 1] - ref[1];
    if (Number.isFinite(dx) && Number.isFinite(dy)) { acc += dx * dx + dy * dy; n++; }
  }
  return n ? Math.sqrt(acc / n) : NaN;
}

export function geoRadius(pts: ArrayLike<number>, ref: [number, number]): number {
  let r = 0;
  for (let i = 0; i < pts.length; i += 2) {
    const dx = pts[i] - ref[0], dy = pts[i + 1] - ref[1];
    if (Number.isFinite(dx) && Number.isFinite(dy)) r = Math.max(r, Math.hypot(dx, dy));
  }
  return r;
}

/** Airy 斑直径 mm:2.44·λ·F/#(λ: nm → mm)。 */
export function airyDiameterMm(wavelengthNm: number, fNumber: number): number {
  return 2.44 * wavelengthNm * 1e-6 * fNumber;
}

export function fmtUm(mm: number): string {
  return `${(mm * 1000).toFixed(1)} μm`;
}
