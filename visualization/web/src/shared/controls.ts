/** 顶部控制条:文件打开 / population / field / wavelength chips + 各窗口扩展控件。 */

import { uploadFile, type SystemMeta } from "./api";
import { fieldColor, wavelengthColor } from "./palette";

export class ControlBar {
  pop: number;
  fields: Set<number>;
  wvls: Set<number>;
  onChange: (what: "pop" | "filter" | "extra") => void = () => {};
  private errorEl: HTMLDivElement;

  constructor(
    root: HTMLElement,
    sys: SystemMeta,
    extras: HTMLElement[] = [],
    init: { pop?: number; fields?: Set<number>; wvls?: Set<number> } = {},
  ) {
    this.pop = Math.min(sys.population - 1, Math.max(0, init.pop ?? 0));
    this.fields = init.fields ?? new Set(sys.fields_deg.map((_, i) => i));
    this.wvls = init.wvls ?? new Set(sys.wavelengths_nm.map((_, i) => i));
    root.className = "control-bar";

    // 文件打开(换文件即整页重载,状态经 URL hash 保留)
    root.append(makeOpenButton(() => location.reload()), sep());

    // population:滑块 + 数字输入
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = String(sys.population - 1);
    slider.value = String(this.pop);
    const num = document.createElement("input");
    num.type = "number";
    num.min = "0";
    num.max = String(sys.population - 1);
    num.value = String(this.pop);
    const onPop = (v: string) => {
      const p = Math.round(Number(v));
      if (!Number.isFinite(p)) return;
      this.pop = Math.min(sys.population - 1, Math.max(0, p));
      slider.value = num.value = String(this.pop);
      this.onChange("pop");
    };
    slider.addEventListener("input", () => onPop(slider.value));
    num.addEventListener("change", () => onPop(num.value));
    root.append(labelEl("Population"), slider, num, sep());

    sys.fields_deg.forEach(([fx, fy], i) => {
      root.append(
        this.chip(`F${i} (${fx.toFixed(1)}°, ${fy.toFixed(1)}°)`, fieldColor(i),
          this.fields.has(i), (on) => {
            on ? this.fields.add(i) : this.fields.delete(i);
            this.onChange("filter");
          }),
      );
    });
    root.append(sep());
    sys.wavelengths_nm.forEach((nm, i) => {
      root.append(
        this.chip(`${nm.toFixed(0)} nm`, wavelengthColor(nm), this.wvls.has(i), (on) => {
          on ? this.wvls.add(i) : this.wvls.delete(i);
          this.onChange("filter");
        }),
      );
    });
    if (extras.length) root.append(sep(), ...extras);

    this.errorEl = document.createElement("div");
    this.errorEl.className = "error-bar";
    this.errorEl.style.display = "none";
    root.after(this.errorEl);
  }

  private chip(text: string, color: string, on: boolean, cb: (on: boolean) => void): HTMLElement {
    const b = document.createElement("button");
    b.className = `chip${on ? "" : " off"}`;
    const dot = document.createElement("i");
    dot.style.background = color;
    b.append(dot, document.createTextNode(text));
    b.addEventListener("click", () => {
      const now = b.classList.toggle("off");
      cb(!now);
    });
    return b;
  }

  showError(msg: string): void {
    this.errorEl.textContent = msg;
    this.errorEl.style.display = "";
  }

  clearError(): void {
    this.errorEl.style.display = "none";
  }
}

/** 文件打开按钮:.toml / .pth 上传建系统,成功后回调(失败时 alert)。 */
export function makeOpenButton(onLoaded: () => void): HTMLElement {
  const label = document.createElement("label");
  label.className = "open-btn";
  label.textContent = "打开 .toml / .pth";
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".toml,.pth";
  input.hidden = true;
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (!file) return;
    label.textContent = `载入 ${file.name} …`;
    uploadFile(file)
      .then(onLoaded)
      .catch((e: unknown) => {
        label.textContent = "打开 .toml / .pth";
        alert(`载入失败:${e instanceof Error ? e.message : String(e)}`);
      });
  });
  label.appendChild(input);
  return label;
}

/** 未加载系统时的整页提示(打开按钮 + 成功后回调)。 */
export function showEmpty(host: HTMLElement, onLoaded: () => void): void {
  const div = document.createElement("div");
  div.className = "empty";
  const hint = document.createElement("div");
  hint.textContent = "尚未加载光学系统";
  div.append(hint, makeOpenButton(onLoaded));
  host.appendChild(div);
}

/** 文本标签。 */
export function labelEl(text: string): HTMLElement {
  const s = document.createElement("span");
  s.className = "ctl-label";
  s.textContent = text;
  return s;
}

function sep(): HTMLElement {
  const s = document.createElement("span");
  s.className = "ctl-sep";
  return s;
}

/** 下拉选择控件(扩展控件用)。 */
export function makeSelect(
  text: string,
  options: [string, string][],
  initial: string,
  cb: (v: string) => void,
): HTMLElement {
  const wrap = document.createElement("span");
  wrap.className = "ctl-select";
  const sel = document.createElement("select");
  for (const [v, l] of options) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = l;
    sel.appendChild(o);
  }
  sel.value = initial;
  sel.addEventListener("change", () => cb(sel.value));
  wrap.append(labelEl(text), sel);
  return wrap;
}

/** 开关控件(扩展控件用)。 */
export function makeToggle(text: string, initial: boolean, cb: (on: boolean) => void): HTMLElement {
  const wrap = document.createElement("label");
  wrap.className = "ctl-toggle";
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = initial;
  box.addEventListener("change", () => cb(box.checked));
  wrap.append(box, document.createTextNode(text));
  return wrap;
}

/** 数字输入控件(扩展控件用)。 */
export function makeNumber(text: string, initial: string, cb: (v: string) => void): HTMLElement {
  const wrap = document.createElement("span");
  wrap.className = "ctl-select";
  const input = document.createElement("input");
  input.type = "number";
  input.step = "any";
  input.value = initial;
  input.addEventListener("change", () => cb(input.value));
  wrap.append(labelEl(text), input);
  return wrap;
}

/** URL hash 状态(#pop=3&fields=0,2&wvls=1)。 */
export function readHash(): URLSearchParams {
  return new URLSearchParams(location.hash.slice(1));
}

export function writeHash(patch: Record<string, string>): void {
  const h = readHash();
  for (const [k, v] of Object.entries(patch)) h.set(k, v);
  history.replaceState(null, "", `#${h}`);
}
