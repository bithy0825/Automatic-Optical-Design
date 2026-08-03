import * as THREE from "three";
import type { Packet, SpotMeta } from "../shared/api";
import { wavelengthColor } from "../shared/palette";
import { airyDiameterMm, centroid, fmtUm, geoRadius, rmsRadius } from "./annotate";

export interface CellOpts {
  reference: "chief" | "centroid";
  showDead: boolean;
  fNumber: number | null;
}

/** 一格的纯数据准备(与渲染无关;统一标尺需先算完全部格子的 r)。 */
export interface CellPrep {
  alive: number[];
  dead: number[];
  ref: [number, number];
  r: number;
  airyR: number;
  wlNm: number;
  stats: string;
}

export function prepCell(
  packet: Packet<SpotMeta>,
  fi: number,
  wi: number,
  opts: CellOpts,
): CellPrep {
  const spots = packet.sections.spots;
  const [, W, N] = spots.shape as [number, number, number, number];
  const S = spots.data as Float32Array;
  const H = packet.sections.holds.data as Uint8Array;
  const ci = packet.meta.chief_index;
  const base = (fi * W + wi) * N;
  const alive: number[] = [];
  const dead: number[] = [];
  for (let n = 0; n < N; n++) {
    const x = S[(base + n) * 2], y = S[(base + n) * 2 + 1];
    (H[base + n] ? alive : dead).push(x, y);
  }
  const chief: [number, number] = [S[(base + ci) * 2], S[(base + ci) * 2 + 1]];
  const ref = opts.reference === "chief" ? chief : centroid(alive);
  const rms = rmsRadius(alive, ref);
  const geo = geoRadius(alive, ref);
  const wlNm = packet.meta.wavelengths_nm[wi];
  const airyR = opts.fNumber ? airyDiameterMm(wlNm, opts.fNumber) / 2 : 0;
  const r = Math.max(geo, airyR, 1e-6) * 1.6; // 余量:点云约占半幅 62%
  const lines: string[] = alive.length
    ? [`RMS ${fmtUm(rms)}`, `GEO ${fmtUm(geo)}`]
    : ["无存活光线"];
  if (airyR > 0) lines.push(`AIRY ${fmtUm(airyR)}`); // 衍射极限半径(本格波长)
  return {
    alive, dead, ref, r, airyR, wlNm,
    stats: lines.join("\n"),
  };
}

/** 每格独立的渲染实体:自己的 canvas + renderer + scene + camera。
 *  坐标映射为零——canvas 即格子,尺寸自洽,与页面 dpr/缩放无关。 */
export interface CellView {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.OrthographicCamera;
}

export function createCellView(canvas: HTMLCanvasElement): CellView {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x0f1115);
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
  camera.position.z = 5;
  return { renderer, scene: new THREE.Scene(), camera };
}

export function disposeCellView(v: CellView): void {
  clearScene(v.scene);
  v.renderer.dispose();
}

function clearScene(scene: THREE.Scene): void {
  for (const child of [...scene.children]) {
    scene.remove(child);
    const anyC = child as THREE.Points | THREE.LineSegments;
    anyC.geometry?.dispose();
    (anyC.material as THREE.Material | undefined)?.dispose();
  }
}

/** 把一格画进它自己的 canvas;r 为视野半径(统一标尺时传共享值)。 */
export function renderCell(v: CellView, p: CellPrep, r: number, showDead: boolean): void {
  const host = v.renderer.domElement.parentElement!;
  const w = host.clientWidth, h = host.clientHeight;
  if (w === 0 || h === 0) return;
  v.renderer.setSize(w, h, false);
  clearScene(v.scene);

  const [cx, cy] = p.ref;
  const at = (x: number, y: number): [number, number, number] => [x - cx, y - cy, 0];

  // 死光线(灰,压在存活点下面)
  if (showDead && p.dead.length) {
    const pos: number[] = [];
    for (let i = 0; i < p.dead.length; i += 2) {
      const [x, y, z] = at(p.dead[i], p.dead[i + 1]);
      if (Number.isFinite(x) && Number.isFinite(y)) pos.push(x, y, z);
    }
    v.scene.add(points(pos, "#666666", 2, 0.4, 1));
  }
  // 存活点(波长着色)
  if (p.alive.length) {
    const pos: number[] = [];
    for (let i = 0; i < p.alive.length; i += 2) {
      const [x, y, z] = at(p.alive[i], p.alive[i + 1]);
      if (Number.isFinite(x) && Number.isFinite(y)) pos.push(x, y, z);
    }
    v.scene.add(points(pos, wavelengthColor(p.wlNm), 3, 1, 2));
  }
  // Airy 圆
  if (p.airyR > 0) {
    const seg = 64;
    const circ: number[] = [];
    for (let i = 0; i < seg; i++) {
      const t = (i / seg) * Math.PI * 2;
      circ.push(Math.cos(t) * p.airyR, Math.sin(t) * p.airyR, 0);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(circ, 3));
    const loop = new THREE.LineLoop(g,
      new THREE.LineBasicMaterial({
        color: 0x8b93a3, transparent: true, opacity: 0.7,
        depthTest: false, depthWrite: false,
      }));
    loop.frustumCulled = false;
    loop.renderOrder = 3;
    v.scene.add(loop);
  }
  // 参考十字
  const a = r * 0.08;
  const cross = new THREE.BufferGeometry();
  cross.setAttribute("position",
    new THREE.Float32BufferAttribute([-a, 0, 0, a, 0, 0, 0, -a, 0, 0, a, 0], 3));
  const cl = new THREE.LineSegments(cross,
    new THREE.LineBasicMaterial({ color: 0xd7dbe2, depthTest: false, depthWrite: false }));
  cl.frustumCulled = false;
  cl.renderOrder = 4;
  v.scene.add(cl);

  // 相机:以参考点为中心、半径 r(按格子宽高比扩展)
  const aspect = w / Math.max(h, 1);
  const hw = r * Math.max(1, aspect), hh = r * Math.max(1, 1 / aspect);
  v.camera.left = -hw; v.camera.right = hw;
  v.camera.top = hh; v.camera.bottom = -hh;
  v.camera.updateProjectionMatrix();
  v.renderer.render(v.scene, v.camera);
}

function points(
  pos: number[],
  color: string,
  size: number,
  opacity: number,
  renderOrder: number,
): THREE.Points {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  const p = new THREE.Points(
    g,
    new THREE.PointsMaterial({
      color, size, sizeAttenuation: false,
      transparent: opacity < 1, opacity,
      depthTest: false, depthWrite: false,
    }),
  );
  p.frustumCulled = false;
  p.renderOrder = renderOrder;
  return p;
}
