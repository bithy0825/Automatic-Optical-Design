import * as THREE from "three";
import type { LayoutMeta, Tensor } from "../shared/api";
import { materialColor } from "../shared/palette";

export interface LensGeometry {
  group: THREE.Group;
  bounds: { x0: number; y0: number; x1: number; y1: number };
}

const isAir = (name: string) => name === "" || name.toLowerCase() === "air";

/** 材料链分段 → 透镜实体(含胶合界面)+ 未配对轮廓线。
 *  regions[j]:j=0 为首面上游;j>=1 为面 j-1 的下游介质名。 */
export function buildLenses(meta: LayoutMeta, profiles: Tensor, rims: Tensor): LensGeometry {
  const [S, NP] = profiles.shape as [number, number, number];
  const D = profiles.data as Float32Array;
  const group = new THREE.Group();
  const b = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
  const eat = (z: number, x: number) => {
    b.x0 = Math.min(b.x0, z); b.x1 = Math.max(b.x1, z);
    b.y0 = Math.min(b.y0, x); b.y1 = Math.max(b.y1, x);
  };

  /** 面 s 的 profile → (z, x) 点列(丢 NaN)。 */
  const profAt = (s: number): [number, number][] => {
    const out: [number, number][] = [];
    for (let i = 0; i < NP; i++) {
      const x = D[(s * NP + i) * 2], z = D[(s * NP + i) * 2 + 1];
      if (Number.isFinite(x) && Number.isFinite(z)) {
        out.push([z, x]);
        eat(z, x);
      }
    }
    return out;
  };

  let s = 0;
  while (s < S) {
    const opensBody = !isAir(meta.regions[s + 1] ?? "") && isAir(meta.regions[s]);
    if (opensBody) {
      let e = s;
      while (e + 1 < S && !isAir(meta.regions[e + 1])) e++;
      // 注意:e+1 面与 e 面之间的介质是 regions[e+1](面 e 的下游)
      group.add(lensBody(profAt, s, e, meta.regions[s + 1]));
      for (let i = s + 1; i < e; i++) group.add(profileLine(profAt(i), 0x8f98a8)); // 胶合界面
      s = e + 1;
    } else {
      group.add(profileLine(profAt(s), 0xcfd6e4)); // 未配对(sensor / 空气间隔面)
      s++;
    }
  }
  if (!Number.isFinite(b.x0)) Object.assign(b, { x0: -1, y0: -1, x1: 1, y1: 1 });
  return { group, bounds: b };
}

/** 闭合透镜体:前表面(下→上)+ 边缘 + 后表面(上→下),半透明填充 + 轮廓。 */
function lensBody(
  profAt: (s: number) => [number, number][],
  a: number,
  b: number,
  glass: string,
): THREE.Group {
  const front = profAt(a);
  const back = profAt(b);
  const g = new THREE.Group();
  if (!front.length || !back.length) return g;

  const shape = new THREE.Shape();
  shape.moveTo(front[0][0], front[0][1]);
  for (const [z, x] of front.slice(1)) shape.lineTo(z, x);
  for (const [z, x] of [...back].reverse()) shape.lineTo(z, x);
  shape.closePath();

  const fill = new THREE.Mesh(
    new THREE.ShapeGeometry(shape),
    new THREE.MeshBasicMaterial({
      color: materialColor(glass),
      transparent: true,
      opacity: 0.22,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
    }),
  );
  fill.frustumCulled = false;
  fill.renderOrder = 1;
  g.add(fill, profileLine(front, 0xcfd6e4), profileLine(back, 0xcfd6e4));
  return g;
}

function profileLine(pts: [number, number][], color: number): THREE.Line {
  const g = new THREE.BufferGeometry();
  g.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(pts.flatMap(([z, x]) => [z, x, 0]), 3),
  );
  const l = new THREE.Line(
    g,
    new THREE.LineBasicMaterial({ color, depthTest: false, depthWrite: false }),
  );
  l.frustumCulled = false;
  l.position.z = 0.05;
  l.renderOrder = 2;
  return l;
}
