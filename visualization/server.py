"""本地 HTTP 服务：单页前端 + .toml/.pth 文件加载 + 按需追迹（仅标准库）。

* ``GET /``                  —— 单页应用（同目录 ``page.html``，无构建步骤）。
* ``POST /api/load?name=…``  —— 上传 .toml / .pth 内容，建系统并返回元数据。
* ``GET  /api/layout``       —— 光路图数据（``pop`` / ``rays`` 参数）。
* ``GET  /api/spot``         —— 点列图数据（``pop`` / ``density`` / ``sampling``）。
"""

import json
import math
import tomllib
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch

from component import Gap, Refractor, Sensor, Sequential, Stop
from core import term
from optimization import build_sequential, build_target, total_loss
from optimization.target import Target
from visualization.trace import (
    LayoutData,
    SpotData,
    probe_illumination,
    trace_layout,
    trace_spot,
)

_PAGE = Path(__file__).parent / "page.html"


class _Error(Exception):
    """HTTP 错误（status + message）。"""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _jsonable(obj: Any) -> Any:
    """张量/嵌套列表 → JSON 可序列化结构（非有限值 → None）。"""
    if isinstance(obj, torch.Tensor):
        return _jsonable(obj.tolist())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


class _Store:
    """当前载入的系统与按需追迹缓存（单文档模型）。"""

    def __init__(self, seq: Sequential, target: Target | None, blocks) -> None:
        self.seq = seq.to(torch.float64)  # 视图统一 f64；不动调用方的系统
        self.target = target
        self.blocks = blocks
        self._layout: dict[int, LayoutData] = {}
        self._spot: dict[tuple[int, str], SpotData] = {}
        self._losses: dict[str, list[float]] | None = None

    def layout(self, n_rays: int) -> LayoutData:
        if n_rays not in self._layout:
            self._layout[n_rays] = trace_layout(self.seq, n_rays)
        return self._layout[n_rays]

    def spot(self, density: int, sampling: str) -> SpotData:
        key = (density, sampling)
        if key not in self._spot:
            self._spot[key] = trace_spot(self.seq, density, sampling)
        return self._spot[key]

    def losses(self) -> dict[str, list[float]] | None:
        """逐分项损失（原始采样口径，一次性缓存）；无 target/blocks 时为 None。"""
        if self._losses is None and self.target is not None and self.blocks is not None:
            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(self.seq.dtype)
            try:
                with torch.no_grad():
                    flow = self.seq()
                    total, parts = total_loss(flow, self.seq, self.target, self.blocks)
            finally:
                torch.set_default_dtype(prev_dtype)
            self._losses = {
                **{k: v.tolist() for k, v in parts.items()},
                "total": total.tolist(),
            }
        return self._losses

    def total_length(self) -> list[float]:
        """系统总长：首个折射面顶点 → 传感器（其间的 gap 厚度之和，逐 pop）。"""
        P = self.seq.population
        total = torch.zeros(P, device=self.seq.device, dtype=self.seq.dtype)
        seen_refractor = False
        for comp in self.seq:
            if isinstance(comp, Refractor):
                seen_refractor = True
            elif isinstance(comp, Gap) and seen_refractor:
                total = total.add(comp.Thickness.detach().view(P, -1).sum(dim=1))
        return total.tolist()


def _load_payload(name: str, blob: bytes) -> tuple[Sequential, Target | None, Any]:
    """上传内容 → 光学系统：``.pth`` 为训练存档，``.toml`` 为配置。"""
    if name.endswith(".pth"):
        cfg, state = torch.load(BytesIO(blob), weights_only=True, map_location="cpu")
        target = build_target(cfg)
        seq = build_sequential(cfg, target)
        seq.load_state_dict(state, strict=True)
        return seq, target, term.COMPONENT.resolve(cfg)
    if name.endswith(".toml"):
        cfg = tomllib.loads(blob.decode("utf-8"))
        return build_sequential(cfg), build_target(cfg), term.COMPONENT.resolve(cfg)
    raise _Error(400, f"unsupported file type: {name!r} (expected .toml or .pth)")


def _system_meta(store: _Store, name: str) -> dict[str, Any]:
    """载入应答：种群、照明、面清单、目标规格、逐 pop 损失。"""
    src = store.seq[0]
    fields_deg, wls = probe_illumination(store.seq)
    surfaces: list[dict] = []
    n = 0
    for comp in store.seq:
        if isinstance(comp, Refractor):
            n += 1
            surfaces.append({"label": f"S{n}", "kind": str(comp.shape.kind.canonical)})
        elif isinstance(comp, Stop):
            surfaces.append({"label": "Stop", "kind": "stop"})
        elif isinstance(comp, Sensor):
            surfaces.append({"label": "Sensor", "kind": "sensor"})
    meta: dict[str, Any] = {
        "name": name,
        "population": src.population,
        "epd": src.epd,
        "fields_deg": fields_deg.tolist(),
        "wavelengths_nm": wls.tolist(),
        "surfaces": surfaces,
        "total_length": store.total_length(),
        "target": None
        if store.target is None
        else {
            "id": store.target.id,
            "F": store.target.F,
            "effl": store.target.effl,
        },
    }
    losses = store.losses()
    if losses is not None:
        meta["losses"] = losses
    return meta


def _int_param(qs: dict[str, list[str]], name: str, default: int, *, lo: int, hi: int) -> int:
    raw = qs.get(name, [str(default)])[0]
    try:
        value = int(raw)
    except ValueError:
        raise _Error(400, f"{name} must be an integer, got {raw!r}") from None
    if not lo <= value <= hi:
        raise _Error(400, f"{name} must be in [{lo}, {hi}], got {value}")
    return value


def create_server(port: int = 8000) -> ThreadingHTTPServer:
    """构建服务实例（不启动）；port=0 时由系统分配端口（测试用）。"""
    store: dict[str, _Store] = {}  # 单文档：载入即替换

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                self._route_get()
            except _Error as e:
                self._json({"error": e.message}, e.status)
            except (ValueError, KeyError, IndexError) as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 400)
            except Exception as e:  # 服务边界统一兜底
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self) -> None:
            try:
                self._route_post()
            except _Error as e:
                self._json({"error": e.message}, e.status)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _route_get(self) -> None:
            url = urlparse(self.path)
            qs = parse_qs(url.query)
            if url.path == "/":
                self._page()
                return
            if "current" not in store:
                raise _Error(400, "no document loaded (open a .toml or .pth first)")
            st = store["current"]
            P = st.seq.population
            match url.path:
                case "/api/layout":
                    pop = _int_param(qs, "pop", 0, lo=0, hi=P - 1)
                    rays = _int_param(qs, "rays", 21, lo=2, hi=101)
                    self._json(_layout_packet(st, pop, rays))
                case "/api/spot":
                    pop = _int_param(qs, "pop", 0, lo=0, hi=P - 1)
                    density = _int_param(qs, "density", 16, lo=2, hi=64)
                    sampling = qs.get("sampling", ["uniform"])[0]
                    if sampling not in ("uniform", "fibonacci"):
                        raise _Error(400, f"sampling must be uniform|fibonacci, got {sampling!r}")
                    self._json(_spot_packet(st, pop, density, sampling))
                case _:
                    raise _Error(404, f"not found: {url.path}")

        def _route_post(self) -> None:
            url = urlparse(self.path)
            if url.path != "/api/load":
                raise _Error(404, f"not found: {url.path}")
            name = parse_qs(url.query).get("name", [""])[0]
            blob = self.rfile.read(int(self.headers["Content-Length"]))
            seq, target, blocks = _load_payload(name, blob)
            store["current"] = _Store(seq, target, blocks)
            self._json(_system_meta(store["current"], name))

        def _json(self, obj: dict, status: int = 200) -> None:
            body = json.dumps(obj, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _page(self) -> None:
            body = _PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _fmt: str, *_args: object) -> None:  # 静默访问日志
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def _layout_packet(st: _Store, pop: int, n_rays: int) -> dict[str, Any]:
    data = st.layout(n_rays)
    return {
        "meta": {
            "labels": data.labels,
            "kinds": data.kinds,
            "regions": [r[pop] for r in data.regions],
            "effl": float(data.effl[pop]),
            "total_length": st.total_length()[pop],
            "fields_deg": data.fields_deg.tolist(),
            "wavelengths_nm": data.wavelengths_nm.tolist(),
        },
        "profiles": _jsonable(data.profiles[pop]),
        "rims": _jsonable(data.rims[pop]),
        "paths": _jsonable(data.paths[pop]),
        "holds": _jsonable(data.holds[pop].to(torch.uint8)),
    }


def _spot_packet(st: _Store, pop: int, density: int, sampling: str) -> dict[str, Any]:
    data = st.spot(density, sampling)
    return {
        "meta": {
            "fields_deg": data.fields_deg.tolist(),
            "wavelengths_nm": data.wavelengths_nm.tolist(),
        },
        "spots": _jsonable(data.spots[pop]),
        "holds": _jsonable(data.holds[pop].to(torch.uint8)),
    }


def serve(port: int = 8000, open_browser: bool = False) -> None:
    """启动可视化服务（阻塞，Ctrl+C 退出）。"""
    server = create_server(port)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"[visualization] serving at {url}  (Ctrl+C 退出)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
