import * as THREE from "three";
import type { Tensor } from "../shared/api";
import { fieldColor, wavelengthColor } from "../shared/palette";

export interface RayFilter {
  fields: ReadonlySet<number>;
  wvls: ReadonlySet<number>;
  colorBy: "field" | "wvl";
  showDead: boolean;
  wavelengths_nm: number[];
}

/** 光线折线 → LineSegments:存活按设定着色;死光线灰色半透明(可关)。
 *  路径在首次 hold=0 的面处截断(该点为死亡点,保留)。 */
export function buildRays(paths: Tensor, holds: Tensor, f: RayFilter): THREE.Group {
  const [F, W, N, K] = paths.shape as [number, number, number, number, number];
  const P = paths.data as Float32Array;
  const H = holds.data as Uint8Array;
  const alive: number[] = [];
  const aliveCol: number[] = [];
  const dead: number[] = [];
  const c = new THREE.Color();
  const pIdx = (fi: number, wi: number, ni: number, k: number) =>
    (((fi * W + wi) * N + ni) * K + k) * 3;
  const hIdx = (fi: number, wi: number, ni: number, k: number) =>
    (((fi * W + wi) * N + ni) * K + k);

  for (let fi = 0; fi < F; fi++) {
    if (!f.fields.has(fi)) continue;
    for (let wi = 0; wi < W; wi++) {
      if (!f.wvls.has(wi)) continue;
      c.set(f.colorBy === "field" ? fieldColor(fi) : wavelengthColor(f.wavelengths_nm[wi]));
      for (let ni = 0; ni < N; ni++) {
        let kmax = K - 1;
        for (let k = 0; k < K; k++) {
          if (!H[hIdx(fi, wi, ni, k)]) { kmax = k; break; }
        }
        const isAlive = H[hIdx(fi, wi, ni, K - 1)] === 1;
        for (let k = 0; k < kmax; k++) {
          const a = pIdx(fi, wi, ni, k), b = pIdx(fi, wi, ni, k + 1);
          const xa = P[a], ya = P[a + 1], za = P[a + 2];
          const xb = P[b], yb = P[b + 1], zb = P[b + 2];
          if (![xa, ya, za, xb, yb, zb].every(Number.isFinite)) continue;
          (isAlive ? alive : dead).push(za, xa, 0, zb, xb, 0);
          if (isAlive) aliveCol.push(c.r, c.g, c.b, c.r, c.g, c.b);
        }
      }
    }
  }
  const group = new THREE.Group();
  if (f.showDead && dead.length) group.add(segments(dead, null, 0.35, 2));
  if (alive.length) group.add(segments(alive, aliveCol, 1, 3));
  return group;
}

function segments(
  pos: number[],
  col: number[] | null,
  opacity: number,
  renderOrder: number,
): THREE.LineSegments {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  if (col) g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  const l = new THREE.LineSegments(
    g,
    new THREE.LineBasicMaterial({
      vertexColors: col !== null,
      color: col ? 0xffffff : 0x888888,
      transparent: opacity < 1,
      opacity,
      depthTest: false,
      depthWrite: false,
    }),
  );
  l.frustumCulled = false;
  l.position.z = 0.1; // 2D 共面分层:光线在透镜之上,关闭深度测试防 z-fighting
  l.renderOrder = renderOrder;
  return l;
}
