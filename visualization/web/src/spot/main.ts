import { fetchJson, fetchPacket, type Packet, type SpotMeta, type SystemMeta } from "../shared/api";
import { ControlBar, makeSelect, makeToggle, readHash, writeHash } from "../shared/controls";
import { fieldColor, wavelengthColor } from "../shared/palette";
import {
  createCellView, disposeCellView, prepCell, renderCell,
  type CellPrep, type CellView,
} from "./cells";

interface Cell {
  stats: HTMLElement;
  view: CellView;
}

export async function boot(): Promise<void> {
  const sys = await fetchJson<SystemMeta>("/api/system");
  const viewHost = document.getElementById("view")!;
  const h0 = readHash();

  const state = {
    density: Number(h0.get("density")) || 16,
    sampling: (h0.get("sampling") === "fibonacci" ? "fibonacci" : "uniform") as "uniform" | "fibonacci",
    reference: (h0.get("ref") === "centroid" ? "centroid" : "chief") as "chief" | "centroid",
    scale: (h0.get("scale") === "independent" ? "independent" : "shared") as "shared" | "independent",
    showDead: h0.get("dead") !== "0",
  };
  const init = {
    pop: h0.has("pop") ? Number(h0.get("pop")) : undefined,
    fields: h0.has("fields")
      ? new Set(h0.get("fields")!.split(",").filter(Boolean).map(Number))
      : undefined,
    wvls: h0.has("wvls")
      ? new Set(h0.get("wvls")!.split(",").filter(Boolean).map(Number))
      : undefined,
  };

  const gridEl = document.createElement("div");
  gridEl.className = "spot-grid";
  viewHost.appendChild(gridEl);

  let packet: Packet<SpotMeta> | null = null;
  let cells = new Map<string, Cell>();

  /** 逐格渲染:先算全部 prep(统一标尺需要全局最大 r),再各画各的 canvas。 */
  const render = () => {
    if (!packet) return;
    const opts = { reference: state.reference, showDead: state.showDead, fNumber: sys.target?.F ?? null };
    const preps = new Map<string, CellPrep>();
    let rShared = 0;
    for (const key of cells.keys()) {
      const [fi, wi] = key.split(":").map(Number);
      const p = prepCell(packet, fi, wi, opts);
      preps.set(key, p);
      rShared = Math.max(rShared, p.r);
    }
    for (const [key, cell] of cells) {
      const p = preps.get(key)!;
      cell.stats.textContent = p.stats;
      renderCell(cell.view, p, state.scale === "shared" ? rShared : p.r, state.showDead);
    }
  };

  /** 行 = field、列 = wavelength 的格子 DOM;键集不变则复用(canvas/上下文保留)。 */
  function rebuildGrid(): void {
    if (!packet) return;
    const fields = [...bar.fields].sort((a, b) => a - b);
    const wvls = [...bar.wvls].sort((a, b) => a - b);
    const keys = fields.flatMap((fi) => wvls.map((wi) => `${fi}:${wi}`));
    if (keys.length === cells.size && keys.every((k) => cells.has(k))) return;

    for (const cell of cells.values()) disposeCellView(cell.view);
    cells = new Map();
    gridEl.innerHTML = "";
    gridEl.style.gridTemplateColumns = `auto repeat(${Math.max(wvls.length, 1)}, 1fr)`;
    gridEl.style.gridTemplateRows = `auto repeat(${Math.max(fields.length, 1)}, 1fr)`;
    gridEl.appendChild(document.createElement("div")).className = "hdr";
    for (const wi of wvls) {
      const h = document.createElement("div");
      h.className = "hdr";
      h.textContent = `λ ${packet.meta.wavelengths_nm[wi].toFixed(0)} nm`;
      h.style.color = wavelengthColor(packet.meta.wavelengths_nm[wi]);
      gridEl.appendChild(h);
    }
    for (const fi of fields) {
      const h = document.createElement("div");
      h.className = "hdr";
      const [fx, fy] = packet.meta.fields_deg[fi];
      h.textContent = `F${fi} (${fx.toFixed(1)}°, ${fy.toFixed(1)}°)`;
      h.style.color = fieldColor(fi);
      gridEl.appendChild(h);
      for (const wi of wvls) {
        const c = document.createElement("div");
        c.className = "cell";
        const stats = c.appendChild(document.createElement("div"));
        stats.className = "stats";
        const canvas = c.appendChild(document.createElement("canvas"));
        gridEl.appendChild(c);
        cells.set(`${fi}:${wi}`, { stats, view: createCellView(canvas) });
      }
    }
  }

  async function load(): Promise<void> {
    bar.clearError();
    try {
      packet = await fetchPacket<SpotMeta>(
        `/api/spot?pop=${bar.pop}&density=${state.density}&sampling=${state.sampling}`,
      );
    } catch (e) {
      bar.showError(e instanceof Error ? e.message : String(e));
      return;
    }
    rebuildGrid();
    writeHash({ pop: String(bar.pop), density: String(state.density), sampling: state.sampling });
    render();
  }

  const bar = new ControlBar(
    document.getElementById("controls")!,
    sys,
    [
      makeSelect("采样", [["uniform", "同心环"], ["fibonacci", "Fibonacci"]], state.sampling,
        (v) => { state.sampling = v as "uniform" | "fibonacci"; void load(); }),
      makeSelect("光瞳密度", [["8", "8 环"], ["16", "16 环"], ["24", "24 环"], ["32", "32 环"]],
        String(state.density), (v) => { state.density = Number(v); void load(); }),
      makeSelect("参考", [["chief", "主光线"], ["centroid", "质心"]], state.reference,
        (v) => { state.reference = v as "chief" | "centroid"; writeHash({ ref: v }); render(); }),
      makeSelect("标尺", [["shared", "统一"], ["independent", "各格独立"]], state.scale,
        (v) => { state.scale = v as "shared" | "independent"; writeHash({ scale: v }); render(); }),
      makeToggle("死光线", state.showDead,
        (on) => { state.showDead = on; writeHash({ dead: on ? "1" : "0" }); render(); }),
    ],
    init,
  );

  let popTimer: ReturnType<typeof setTimeout> | null = null;
  bar.onChange = (what) => {
    if (what === "pop") {
      if (popTimer) clearTimeout(popTimer);
      popTimer = setTimeout(() => void load(), 150);
    } else {
      writeHash({ fields: [...bar.fields].join(","), wvls: [...bar.wvls].join(",") });
      rebuildGrid();
      render();
    }
  };

  new ResizeObserver(render).observe(viewHost);
  await load();
}
