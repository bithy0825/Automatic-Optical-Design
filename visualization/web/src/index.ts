import { fetchJson, type SystemMeta } from "./shared/api";

const sys = await fetchJson<SystemMeta>("/api/system");

const fov = sys.target
  ? typeof sys.target.fov === "number"
    ? `${sys.target.fov}°`
    : JSON.stringify(sys.target.fov)
  : "?";
document.getElementById("summary")!.textContent =
  `Population ${sys.population} · ${sys.surfaces.length} 面 · ` +
  `λ = ${sys.wavelengths_nm.map((w) => w.toFixed(0)).join(" / ")} nm · ` +
  `FOV ${fov}` +
  (sys.target ? ` · F/${sys.target.F} · EFFL ${sys.target.effl} mm` : "");

for (const card of document.querySelectorAll<HTMLAnchorElement>("a.card")) {
  card.addEventListener("click", (e) => {
    e.preventDefault();
    window.open(card.href, "_blank", "width=1360,height=860");
  });
}
