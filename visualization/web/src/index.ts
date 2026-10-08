import { fetchJson, uploadFile, type SystemMeta } from "./shared/api";

const summary = document.getElementById("summary")!;
const status = document.getElementById("status")!;
const fileInput = document.getElementById("file") as HTMLInputElement;

function describe(sys: SystemMeta): string {
  const fov = sys.target
    ? typeof sys.target.fov === "number"
      ? `${sys.target.fov}°`
      : JSON.stringify(sys.target.fov)
    : "?";
  return (
    `${sys.name} · Population ${sys.population} · ${sys.surfaces.length} 面 · ` +
    `λ = ${sys.wavelengths_nm.map((w) => w.toFixed(0)).join(" / ")} nm · ` +
    `FOV ${fov}` +
    (sys.target ? ` · F/${sys.target.F} · EFFL ${sys.target.effl} mm` : "")
  );
}

async function refresh(): Promise<void> {
  try {
    const sys = await fetchJson<SystemMeta>("/api/system");
    summary.textContent = describe(sys);
  } catch {
    summary.textContent = "未加载系统——请先打开一个 .toml 或 .pth 文件。";
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  status.textContent = `载入 ${file.name} …`;
  uploadFile(file)
    .then((sys) => {
      status.textContent = "";
      summary.textContent = describe(sys);
    })
    .catch((e: unknown) => {
      status.textContent = `载入失败:${e instanceof Error ? e.message : String(e)}`;
    });
});

void refresh();
