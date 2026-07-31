/** HTML overlay 文字标注层:按 key 复用 div,坐标为屏幕 px(调用方换算)。 */

export interface LabelItem {
  key: string;
  x: number;
  y: number;
  text: string;
  cls?: string;
}

/** 在 host(position:relative)内建绝对定位 overlay 容器。 */
export function createOverlay(host: HTMLElement): HTMLElement {
  host.style.position = "relative";
  const el = document.createElement("div");
  el.style.position = "absolute";
  el.style.inset = "0";
  el.style.pointerEvents = "none";
  el.style.overflow = "hidden";
  host.appendChild(el);
  return el;
}

export class LabelLayer {
  private divs = new Map<string, HTMLDivElement>();

  constructor(private container: HTMLElement) {}

  render(items: LabelItem[]): void {
    const seen = new Set<string>();
    for (const it of items) {
      seen.add(it.key);
      let d = this.divs.get(it.key);
      if (!d) {
        d = document.createElement("div");
        d.style.position = "absolute";
        d.style.transform = "translate(-50%, -50%)";
        d.style.whiteSpace = "pre";
        this.container.appendChild(d);
        this.divs.set(it.key, d);
      }
      d.className = it.cls ?? "";
      d.textContent = it.text;
      d.style.left = `${it.x}px`;
      d.style.top = `${it.y}px`;
    }
    for (const [k, d] of this.divs) {
      if (!seen.has(k)) {
        d.remove();
        this.divs.delete(k);
      }
    }
  }
}
