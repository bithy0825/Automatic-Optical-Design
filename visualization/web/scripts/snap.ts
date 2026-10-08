/** 页面截图工具(裸 CDP,驱动系统 Chrome,无需 playwright)。
 *
 * 前置:启动 headless Chrome ——
 *   chrome --headless=new --remote-debugging-port=9222 \
 *     --user-data-dir=%TEMP%/aod-cdp-profile about:blank
 *
 * 用法:URL='http://127.0.0.1:8765/spot#pop=0' OUT=spot.png DSF=1.5 bun scripts/snap.ts
 * (必须用完整 URL——git-bash 会把 /spot 之类的值做 MSYS 路径转换)
 */

const base = process.env.BASE ?? "http://127.0.0.1:8765";
const url = process.env.URL ?? base + (process.env.PAGE ?? "/");
const out = process.env.OUT ?? "shot.png";
const dsf = Number(process.env.DSF ?? "1");
const cdpPort = Number(process.env.CDP ?? "9222");

let target: { webSocketDebuggerUrl: string } | undefined;
for (let i = 0; i < 30; i++) {
  try {
    const list = (await (await fetch(`http://127.0.0.1:${cdpPort}/json/list`)).json()) as {
      type: string;
      webSocketDebuggerUrl: string;
    }[];
    target = list.find((t) => t.type === "page");
    if (target) break;
  } catch {
    /* Chrome 尚未就绪 */
  }
  await new Promise((r) => setTimeout(r, 500));
}
if (!target) throw new Error("no CDP page target — headless Chrome 未启动?");

const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((res, rej) => {
  ws.onopen = res;
  ws.onerror = rej;
});

let mid = 0;
const pending = new Map<number, (v: never) => void>();
ws.onmessage = (m) => {
  const msg = JSON.parse(String(m.data));
  if (msg.id && pending.has(msg.id)) {
    pending.get(msg.id)!(msg as never);
    pending.delete(msg.id);
  } else if (msg.method === "Runtime.consoleAPICalled") {
    const args = (msg.params.args ?? []).map((a: { value?: unknown }) => a.value).join(" ");
    console.log("[console]", msg.params.type, args);
  } else if (msg.method === "Runtime.exceptionThrown") {
    console.log("[exception]", msg.params.exceptionDetails.exception?.description ?? msg.method);
  }
};

function call<T = Record<string, unknown>>(method: string, params: Record<string, unknown> = {}): Promise<T> {
  const id = ++mid;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((res) => pending.set(id, res as (v: unknown) => void as (v: never) => void));
}

await call("Page.enable");
await call("Runtime.enable");
await call("Emulation.setDeviceMetricsOverride", {
  width: 1360, height: 860, deviceScaleFactor: dsf, mobile: false,
});
const nav = await call<{ error?: { message: string } }>("Page.navigate", { url });
if (nav.error) throw new Error(`navigate failed: ${nav.error.message}`);
// 轮询确认导航真正生效(偶发 navigate 无响应)
for (let i = 0; i < 40; i++) {
  await new Promise((r) => setTimeout(r, 500));
  const r = await call<{ result: { result: { value: string } } }>("Runtime.evaluate", {
    expression: "location.href", returnByValue: true,
  });
  if (r.result?.result?.value?.startsWith(url.split("#")[0])) break;
  if (i === 39) throw new Error(`navigation did not take effect, at ${r.result?.result?.value}`);
}
// 同文档 hash 导航不会重载页面,强制 reload 让 boot 读到目标 hash
await call("Page.reload", { ignoreCache: true });
await new Promise((r) => setTimeout(r, 6000)); // 等追迹数据 + 首帧渲染

const shot = await call<{ result: { data: string } }>("Page.captureScreenshot", { format: "png" });
await Bun.write(out, Uint8Array.from(atob(shot.result.data), (c) => c.charCodeAt(0)));
console.log("saved", out, "dsf =", dsf);
ws.close();

export {};
