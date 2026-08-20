import { fetchJson, fetchPacket, type Packet, type PsfMeta, type SystemMeta } from "../shared/api";
import { ControlBar, makeNumber, makeSelect, makeToggle, readHash, writeHash } from "../shared/controls";
import { fieldColor, wavelengthColor } from "../shared/palette";
import {
  createCellView, disposeCellView, prepCell, renderCell,
  type CellView,
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
    density: Number(h0.get("density")) || 64,
    sampling: (h0.get("sampling") === "uniform" ? "uniform" : "fibonacci") as "uniform" | "fibonacci",
    size: Number(h0.get("size")) || 64,
    deltaUm: Number(h0.get("delta")) || 0,  // μm;0 = 服务器自动
    log: h0.get("log") !== "0",
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
  gridEl.className = "spot-grid";  // 复用点列图网格样式
  viewHost.appendChild(gridEl);

  let packet: Packet<PsfMeta> | null = null;
  let cells = new Map<string, Cell>();

  const render = () => {
    if (!packet) return;
    for (const [key, cell] of cells) {
      const [fi, wi] = key.split(":").map(Number);
      const p = prepCell(packet, fi, wi);
      cell.stats.textContent = p.stats;
      renderCell(cell.view, p, state.log);
    }
  };

  /** 行 = field、列 = wavelength 的格子 DOM;键集不变则复用。 */
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
        canvas.className = "pix";
        gridEl.appendChild(c);
        cells.set(`${fi}:${wi}`, { stats, view: createCellView(canvas) });
      }
    }
  }

  async function load(): Promise<void> {
    bar.clearError();
    try {
      packet = await fetchPacket<PsfMeta>(
        `/api/psf?pop=${bar.pop}&density=${state.density}&sampling=${state.sampling}` +
        `&size=${state.size}&delta=${state.deltaUm / 1000}`,
      );
    } catch (e) {
      bar.showError(e instanceof Error ? e.message : String(e));
      return;
    }
    rebuildGrid();
    writeHash({ pop: String(bar.pop), density: String(state.density),
                sampling: state.sampling, size: String(state.size),
                delta: String(state.deltaUm) });
    render();
    if (packet.meta.warnings.length) {
      bar.showError(`警告: ${packet.meta.warnings.join("; ")}`);
    }
  }

  const bar = new ControlBar(
    document.getElementById("controls")!,
    sys,
    [
      makeSelect("采样", [["fibonacci", "Fibonacci"], ["uniform", "同心环"]], state.sampling,
        (v) => { state.sampling = v as "uniform" | "fibonacci"; void load(); }),
      makeSelect("光瞳密度", [["32", "32 环"], ["64", "64 环"], ["128", "128 环"]],
        String(state.density), (v) => { state.density = Number(v); void load(); }),
      makeSelect("网格", [["64", "64²"], ["128", "128²"], ["256", "256²"]],
        String(state.size), (v) => { state.size = Number(v); void load(); }),
      makeNumber("Δ μm (0=自动)", String(state.deltaUm),
        (v) => { state.deltaUm = Number(v) || 0; void load(); }),
      makeToggle("log", state.log,
        (on) => { state.log = on; writeHash({ log: on ? "1" : "0" }); render(); }),
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
