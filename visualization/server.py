"""本地 HTTP 服务:单源托管静态前端与 /api 追迹数据(仅标准库)。"""

from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from component import Refractor, Sensor, Sequential
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


def system_meta(seq: Sequential, target: Target | None) -> dict:
    """/api/system 元数据:种群、照明、面清单、目标规格。"""
    src = seq[0]
    fields_deg, wls = probe_illumination(seq)
    surfaces: list[dict] = []
    n = 0
    for i, comp in enumerate(seq):
        if isinstance(comp, Refractor):
            n += 1
            surfaces.append({"index": i, "label": f"S{n}",
                             "kind": str(comp.shape.kind.canonical)})
        elif isinstance(comp, Sensor):
            surfaces.append({"index": i, "label": "Sensor", "kind": "sensor"})
    return {
        "population": src.population,
        "epd": src.epd,
        "fields_deg": fields_deg.tolist(),
        "wavelengths_nm": wls.tolist(),
        "surfaces": surfaces,
        "target": None if target is None else {
            "id": target.id, "fov": target.fov, "F": target.F, "effl": target.effl,
        },
    }


class _Error(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _int_param(qs: dict[str, list[str]], name: str, default: int, *, lo: int, hi: int) -> int:
    raw = qs.get(name, [str(default)])[0]
    try:
        value = int(raw)
    except ValueError:
        raise _Error(400, f"{name} must be an integer, got {raw!r}") from None
    if not lo <= value <= hi:
        raise _Error(400, f"{name} must be in [{lo}, {hi}], got {value}")
    return value


def create_server(
    seq: Sequential,
    *,
    target: Target | None = None,
    port: int = 8000,
    web_root: Path = _WEB_ROOT,
) -> ThreadingHTTPServer:
    """构建服务实例(不启动);port=0 时由系统分配端口(测试用)。"""
    cache = TraceCache(seq)
    meta = system_meta(seq, target)
    P = int(meta["population"])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                self._route()
            except _Error as e:
                self._json({"error": e.message}, e.status)
            except PopOutOfRange as e:
                self._json({"error": str(e)}, 400)
            except Exception as e:  # 服务边界统一兜底
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _route(self) -> None:
            url = urlparse(self.path)
            qs = parse_qs(url.query)
            match url.path:
                case "/api/system":
                    self._json(meta)
                case "/api/layout":
                    pop = _int_param(qs, "pop", 0, lo=0, hi=P - 1)
                    rays = _int_param(qs, "rays", 21, lo=2, hi=101)
                    self._binary(cache.layout_packet(pop, rays))
                case "/api/spot":
                    pop = _int_param(qs, "pop", 0, lo=0, hi=P - 1)
                    density = _int_param(qs, "density", 16, lo=2, hi=64)
                    sampling = qs.get("sampling", ["uniform"])[0]
                    if sampling not in ("uniform", "fibonacci"):
                        raise _Error(400, f"sampling must be uniform|fibonacci, got {sampling!r}")
                    self._binary(cache.spot_packet(pop, density, sampling))
                case "/":
                    self._static("index.html")
                case "/layout":
                    self._static("layout.html")
                case "/spot":
                    self._static("spot.html")
                case p:
                    self._static(p.lstrip("/"))

        def _json(self, obj: dict, status: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
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
                "Content-Type", _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")  # 开发期避免陈旧 dist 缓存
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _fmt: str, *_args: object) -> None:  # 静默访问日志
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(
    seq: Sequential,
    *,
    target: Target | None = None,
    port: int = 8000,
    open_browser: bool = False,
) -> None:
    """启动可视化服务(阻塞,Ctrl+C 退出)。"""
    server = create_server(seq, target=target, port=port)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    if not (_WEB_ROOT / "dist").is_dir():
        print("[visualization] web/dist 不存在,请先在 visualization/web 下执行 bun run build")
    print(f"[visualization] serving at {url}  (Ctrl+C 退出)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
