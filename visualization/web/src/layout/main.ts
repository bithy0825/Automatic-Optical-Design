import * as THREE from "three";
import {
  fetchJson, fetchPacket,
  type LayoutMeta, type Packet, type SystemMeta,
} from "../shared/api";
import { ControlBar, makeSelect, makeToggle, readHash, writeHash } from "../shared/controls";
import { LabelLayer, createOverlay, type LabelItem } from "../shared/labels";
import { Viewport, disposeGroup } from "../shared/viewport";
import { buildLenses } from "./lenses";
import { buildRays } from "./rays";

export async function boot(): Promise<void> {
  const sys = await fetchJson<SystemMeta>("/api/system");
  const viewHost = document.getElementById("view")!;
  const h0 = readHash();

  const state = {
    rays: Number(h0.get("rays")) || 21,
    colorBy: (h0.get("color") === "wvl" ? "wvl" : "field") as "field" | "wvl",
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

  // ── three.js 场景 ──
  const canvas = document.createElement("canvas");
  canvas.className = "view";
  viewHost.prepend(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x0f1115);
  const scene = new THREE.Scene();
  const viewport = new Viewport();
  scene.add(viewport.grid);
  const labels = new LabelLayer(createOverlay(viewHost));

  let lensGroup: THREE.Group | null = null;
  let rayGroup: THREE.Group | null = null;
  let packet: Packet<LayoutMeta> | null = null;
  let lensBounds: { x0: number; y0: number; x1: number; y1: number } | null = null;
  let fitted = false;

  const render = () => {
    renderer.setSize(viewHost.clientWidth, viewHost.clientHeight, false);
    renderer.render(scene, viewport.camera);
    updateLabels();
  };

  const updateLabels = () => {
    const items: LabelItem[] = [
      { key: "scale", x: 56, y: viewHost.clientHeight - 18, text: `Δ ${viewport.gridStep} mm`, cls: "lbl" },
    ];
    labels.render(items);
  };

  // ── 数据加载与重建 ──
  async function load(): Promise<void> {
    bar.clearError();
    try {
      packet = await fetchPacket<LayoutMeta>(`/api/layout?pop=${bar.pop}&rays=${state.rays}`);
    } catch (e) {
      bar.showError(e instanceof Error ? e.message : String(e));
      return;
    }
    if (lensGroup) { scene.remove(lensGroup); disposeGroup(lensGroup); }
    const built = buildLenses(packet.meta, packet.sections.profiles, packet.sections.rims);
    lensGroup = built.group;
    lensBounds = built.bounds;
    scene.add(lensGroup);
    rebuildRays();
    if (!fitted && lensBounds) {
      viewport.fit(lensBounds.x0, lensBounds.y0, lensBounds.x1, lensBounds.y1);
      fitted = true;
    }
    writeHash({ pop: String(bar.pop), rays: String(state.rays) });
    render();
  }

  function rebuildRays(): void {
    if (!packet) return;
    if (rayGroup) { scene.remove(rayGroup); disposeGroup(rayGroup); }
    rayGroup = buildRays(packet.sections.paths, packet.sections.holds, {
      fields: bar.fields,
      wvls: bar.wvls,
      colorBy: state.colorBy,
      showDead: state.showDead,
      wavelengths_nm: packet.meta.wavelengths_nm,
    });
    scene.add(rayGroup);
    render();
  }

  // ── 控制条 ──
  const bar = new ControlBar(
    document.getElementById("controls")!,
    sys,
    [
      makeSelect("光线/视场", [["9", "9"], ["15", "15"], ["21", "21"], ["31", "31"]],
        String(state.rays), (v) => { state.rays = Number(v); void load(); }),
      makeSelect("着色", [["field", "按视场"], ["wvl", "按波长"]], state.colorBy,
        (v) => { state.colorBy = v as "field" | "wvl"; writeHash({ color: v }); rebuildRays(); }),
      makeToggle("死光线", state.showDead,
        (on) => { state.showDead = on; writeHash({ dead: on ? "1" : "0" }); rebuildRays(); }),
    ],
    init,
  );

  let popTimer: ReturnType<typeof setTimeout> | null = null;
  bar.onChange = (what) => {
    if (what === "pop") {
      if (popTimer) clearTimeout(popTimer);
      popTimer = setTimeout(() => void load(), 150); // 滑块防抖
    } else {
      writeHash({
        fields: [...bar.fields].join(","),
        wvls: [...bar.wvls].join(","),
      });
      rebuildRays();
    }
  };

  viewport.attach(viewHost, render);
  viewport.apply();
  await load();
}
