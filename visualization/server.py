"""本地 HTTP 服务：静态前端（bun + three.js）+ 文件载入 + 按需追迹（仅标准库）。

* ``GET /``                  —— 启动页（选择 .toml / .pth，进入各视图）。
* ``GET /layout`` ``/spot``  —— 光路图 / 点列图页面（需先 ``bun run build`` 出 web/dist）。
* ``POST /api/load?name=…``  —— 上传 .toml / .pth 内容，建系统并返回元数据。
* ``GET  /api/system``       —— 当前系统元数据（未载入 → 404）。
* ``GET  /api/layout``       —— 光路图二进制帧（``pop`` / ``rays`` 参数）。
* ``GET  /api/spot``         —— 点列图二进制帧（``pop`` / ``density`` / ``sampling``）。
"""

from __future__ import annotations

import json
import tomllib
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch

from component import Refractor, Sensor, Sequential, Stop
from core import term
from optimization import LossWeights, build_sequential, build_target
from optimization.target import Target
from visualization.trace import PopOutOfRange, TraceCache, probe_illumination

_WEB_ROOT = Path(__file__).parent / "web"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
}


class _Error(Exception):
    """HTTP 错误（status + message）。"""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _load_payload(
    name: str, blob: bytes
) -> tuple[Sequential, Target | None, Any, LossWeights | None]:
    """上传内容 → 光学系统：``.pth`` 为训练存档，``.toml`` 为配置。"""
    if name.endswith(".pth"):
        cfg, state = torch.load(BytesIO(blob), weights_only=True, map_location="cpu")
        target = build_target(cfg)
        seq = build_sequential(cfg, target)
        seq.load_state_dict(state, strict=True)
    elif name.endswith(".toml"):
        cfg = tomllib.loads(blob.decode("utf-8"))
        target = build_target(cfg)
        seq = build_sequential(cfg, target)
    else:
        raise _Error(400, f"unsupported file type: {name!r} (expected .toml or .pth)")
    return seq, target, term.COMPONENT.resolve(cfg), LossWeights.from_options(cfg)


def _system_meta(seq: Sequential, target: Target | None, name: str) -> dict[str, Any]:
    """载入应答 / ``/api/system``：种群、照明、面清单、目标规格。"""
    src = seq[0]
    fields_deg, wls = probe_illumination(seq)
    surfaces: list[dict] = []
    n = 0
    for i, comp in enumerate(seq):
        if isinstance(comp, Refractor):
            n += 1
            surfaces.append({"index": i, "label": f"S{n}", "kind": str(comp.shape.kind.canonical)})
        elif isinstance(comp, Stop):
            surfaces.append({"index": i, "label": "Stop", "kind": "stop"})
        elif isinstance(comp, Sensor):
            surfaces.append({"index": i, "label": "Sensor", "kind": "sensor"})
    return {
        "name": name,
        "population": src.population,
        "epd": src.epd,
        "fields_deg": fields_deg.tolist(),
        "wavelengths_nm": wls.tolist(),
        "surfaces": surfaces,
        "target": None
        if target is None
        else {"id": target.id, "fov": target.fov, "F": target.F, "effl": target.effl},
    }


def _int_param(qs: dict[str, list[str]], name: str, default: int, *, lo: int, hi: int) -> int:
    raw = qs.get(name, [str(default)])[0]
    try:
        value = int(raw)
    except ValueError:
        raise _Error(400, f"{name} must be an integer, got {raw!r}") from None
    if not lo <= value <= hi:
        raise _Error(400, f"{name} must be in [{lo}, {hi}], got {value}")
    return value


def _sampling_param(qs: dict[str, list[str]]) -> str:
    sampling = qs.get("sampling", ["uniform"])[0]
    if sampling not in ("uniform", "fibonacci"):
        raise _Error(400, f"sampling must be uniform|fibonacci, got {sampling!r}")
    return sampling


def create_server(port: int = 8000, web_root: Path = _WEB_ROOT) -> ThreadingHTTPServer:
    """构建服务实例（不启动）；port=0 时由系统分配端口（测试用）。"""
    store: dict[str, tuple[TraceCache, dict[str, Any]]] = {}  # 单文档：载入即替换

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                self._route_get()
            except _Error as e:
                self._json({"error": e.message}, e.status)
            except PopOutOfRange as e:
                self._json({"error": str(e)}, 400)
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

        def _current(self) -> tuple[TraceCache, dict[str, Any]]:
            if "current" not in store:
                raise _Error(404, "no document loaded (open a .toml or .pth first)")
            return store["current"]

        def _route_get(self) -> None:
            url = urlparse(self.path)
            qs = parse_qs(url.query)
            match url.path:
                case "/api/system":
                    self._json(self._current()[1])
                case "/api/layout":
                    cache, meta = self._current()
                    P = int(meta["population"])
                    pop = _int_param(qs, "pop", 0, lo=0, hi=P - 1)
                    rays = _int_param(qs, "rays", 21, lo=2, hi=101)
                    self._binary(cache.layout_packet(pop, rays))
                case "/api/spot":
                    cache, meta = self._current()
                    P = int(meta["population"])
                    pop = _int_param(qs, "pop", 0, lo=0, hi=P - 1)
                    density = _int_param(qs, "density", 16, lo=2, hi=64)
                    self._binary(cache.spot_packet(pop, density, _sampling_param(qs)))
                case "/":
                    self._static("index.html")
                case "/layout":
                    self._static("layout.html")
                case "/spot":
                    self._static("spot.html")
                case p:
                    self._static(p.lstrip("/"))

        def _route_post(self) -> None:
            url = urlparse(self.path)
            if url.path != "/api/load":
                raise _Error(404, f"not found: {url.path}")
            name = parse_qs(url.query).get("name", [""])[0]
            blob = self.rfile.read(int(self.headers["Content-Length"]))
            seq, target, blocks, weights = _load_payload(name, blob)
            seq = seq.to(torch.float64)  # 视图统一 f64；不动调用方的系统
            meta = _system_meta(seq, target, name)
            store["current"] = (TraceCache(seq, target, blocks, weights), meta)
            self._json(meta)

        def _json(self, obj: dict, status: int = 200) -> None:
            body = json.dumps(obj, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _binary(self, blob: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def _static(self, rel: str) -> None:
            path = (web_root / rel).resolve()
            if web_root.resolve() not in path.parents or not path.is_file():
                raise _Error(404, f"not found: {rel}")
            body = path.read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                _CONTENT_TYPES.get(path.suffix, "application/octet-stream"),
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")  # 开发期避免陈旧 dist 缓存
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 静默访问日志
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(port: int = 8000, open_browser: bool = False) -> None:
    """启动可视化服务（阻塞，Ctrl+C 退出）。"""
    server = create_server(port)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    if not (_WEB_ROOT / "dist").is_dir():
        print("[visualization] web/dist 不存在，请先在 visualization/web 下执行 bun run build")
    print(f"[visualization] serving at {url}  (Ctrl+C 退出)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
