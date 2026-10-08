import * as THREE from "three";

/** 二维视口:正交相机 + 缩放平移 + 自适应网格(1-2-5 步长)。世界单位 mm。 */

export class Viewport {
  readonly camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
  readonly grid = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0x2a2f3a, depthTest: false, depthWrite: false }),
  );
  center = new THREE.Vector2(0, 0);
  pxPerMm = 50;
  gridStep = 1;
  private el: HTMLElement | null = null;
  private notify: () => void = () => {};

  constructor() {
    this.camera.position.z = 5;
    this.grid.frustumCulled = false;
    this.grid.renderOrder = 0; // 2D 共面分层:网格最底,关闭深度测试防 z-fighting
  }

  /** 绑定宿主元素与交互;视图变化(含尺寸变化)回调 notify。 */
  attach(el: HTMLElement, notify: () => void): void {
    this.el = el;
    this.notify = notify;
    new ResizeObserver(() => this.apply()).observe(el);
    el.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const [wx, wy] = this.screenToWorld(e.offsetX, e.offsetY);
        this.pxPerMm = clamp(this.pxPerMm * Math.exp(-e.deltaY * 0.0012), 0.05, 1e5);
        // 保持光标下的世界点不动
        const [sx, sy] = this.worldToScreen(wx, wy);
        this.center.x += (sx - e.offsetX) / this.pxPerMm;
        this.center.y -= (sy - e.offsetY) / this.pxPerMm;
        this.apply();
      },
      { passive: false },
    );
    let drag: { x: number; y: number } | null = null;
    el.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY };
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener("pointermove", (e) => {
      if (!drag) return;
      this.center.x -= (e.clientX - drag.x) / this.pxPerMm;
      this.center.y += (e.clientY - drag.y) / this.pxPerMm;
      drag = { x: e.clientX, y: e.clientY };
      this.apply();
    });
    el.addEventListener("pointerup", () => (drag = null));
  }

  /** 内容包围盒 → 适配视图(margin 为边距倍率)。 */
  fit(x0: number, y0: number, x1: number, y1: number, margin = 1.15): void {
    if (!this.el) return;
    const w = this.el.clientWidth, h = this.el.clientHeight;
    this.pxPerMm = Math.min(
      w / (Math.max(x1 - x0, 1e-6) * margin),
      h / (Math.max(y1 - y0, 1e-6) * margin),
    );
    this.center.set((x0 + x1) / 2, (y0 + y1) / 2);
    this.apply();
  }

  screenToWorld(sx: number, sy: number): [number, number] {
    const w = this.el!.clientWidth, h = this.el!.clientHeight;
    return [
      this.center.x + (sx - w / 2) / this.pxPerMm,
      this.center.y - (sy - h / 2) / this.pxPerMm,
    ];
  }

  worldToScreen(wx: number, wy: number): [number, number] {
    const w = this.el!.clientWidth, h = this.el!.clientHeight;
    return [
      w / 2 + (wx - this.center.x) * this.pxPerMm,
      h / 2 - (wy - this.center.y) * this.pxPerMm,
    ];
  }

  /** 相机/网格与当前视图参数同步;视图变化后调用(触发 notify)。 */
  apply(): void {
    if (!this.el) return;
    const w = this.el.clientWidth, h = this.el.clientHeight;
    const hw = w / 2 / this.pxPerMm, hh = h / 2 / this.pxPerMm;
    const c = this.camera;
    c.left = this.center.x - hw;
    c.right = this.center.x + hw;
    c.top = this.center.y + hh;
    c.bottom = this.center.y - hh;
    c.updateProjectionMatrix();
    this.rebuildGrid(c.left, c.bottom, c.right, c.top);
    this.notify();
  }

  private rebuildGrid(x0: number, y0: number, x1: number, y1: number): void {
    const step = niceStep((x1 - x0) / 10);
    this.gridStep = step;
    const pos: number[] = [];
    for (let x = Math.ceil(x0 / step) * step; x <= x1; x += step) pos.push(x, y0, 0, x, y1, 0);
    for (let y = Math.ceil(y0 / step) * step; y <= y1; y += step) pos.push(x0, y, 0, x1, y, 0);
    this.grid.geometry.dispose();
    this.grid.geometry = new THREE.BufferGeometry();
    this.grid.geometry.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/** 1-2-5 序列的"好看"步长。 */
export function niceStep(raw: number): number {
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / mag;
  return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * mag;
}

/** 递归释放 Group 内所有 geometry / material。 */
export function disposeGroup(g: THREE.Group): void {
  g.traverse((o) => {
    const anyO = o as THREE.Mesh | THREE.LineSegments | THREE.Points;
    if (anyO.geometry) anyO.geometry.dispose();
    const m = anyO.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(m)) m.forEach((x) => x.dispose());
    else if (m) m.dispose();
  });
}
