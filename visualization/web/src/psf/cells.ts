/** PSF 格渲染:峰值归一热图(inferno)+ Airy 圆 + 中心十字,Canvas 2D。 */

import type { Packet, PsfMeta } from "../shared/api";
import { inferno } from "./colormap";

export interface CellView {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  buf: HTMLCanvasElement;   // 1:1 离屏热图,再最近邻放大
  bctx: CanvasRenderingContext2D;
}

export function createCellView(canvas: HTMLCanvasElement): CellView {
  const buf = document.createElement("canvas");
  return { canvas, ctx: canvas.getContext("2d")!,
           buf, bctx: buf.getContext("2d")! };
}

export function disposeCellView(_v: CellView): void { /* 2D 上下文无需释放 */ }

export interface CellPrep {
  img: Float32Array;  // (H,H),psf[i,j] i↔x j↔y,Σ=1
  h: number;
  peak: number;
  airyPx: number;     // Airy 第一暗环半径(像素),NA 缺失为 0
  stats: string;
}

export function prepCell(packet: Packet<PsfMeta>, fi: number, wi: number): CellPrep {
  const sec = packet.sections.psf;
  const W = sec.shape[1], H = sec.shape[2];
  const off = (fi * W + wi) * H * H;
  const img = (sec.data as Float32Array).slice(off, off + H * H);
  const na = (packet.sections.na.data as Float32Array)[fi * W + wi];
  const lamMm = packet.meta.wavelengths_nm[wi] * 1e-6;
  const airyPx = na > 0 ? (0.61 * lamMm) / na / packet.meta.delta : 0;
  let peak = 0;
  for (let i = 0; i < img.length; i++) if (img[i] > peak) peak = img[i];
  const extentUm = H * packet.meta.delta * 1000;
  return {
    img, h: H, peak, airyPx,
    stats: `pk ${peak.toExponential(2)}\n${extentUm.toFixed(0)} μm`,
  };
}

export function renderCell(v: CellView, p: CellPrep, log: boolean): void {
  const host = v.canvas.parentElement!;
  const wpx = host.clientWidth, hpx = host.clientHeight;
  if (wpx === 0 || hpx === 0) return;
  v.canvas.width = wpx;
  v.canvas.height = hpx;
  const H = p.h, peak = p.peak > 0 ? p.peak : 1;

  // 1:1 离屏:峰值归一 → (可选)3 个数量级 log 压缩 → inferno
  v.buf.width = v.buf.height = H;
  const id = v.bctx.createImageData(H, H);
  for (let x = 0; x < H; x++) {
    for (let y = 0; y < H; y++) {
      const val = p.img[x * H + y] / peak;
      const t = log ? Math.log10(1 + 999 * val) / 3 : val;
      const [r, g, b] = inferno(t);
      const o = (y * H + x) * 4;   // 屏幕行=y 列=x(转置读取)
      id.data[o] = r; id.data[o + 1] = g; id.data[o + 2] = b; id.data[o + 3] = 255;
    }
  }
  v.bctx.putImageData(id, 0, 0);

  const ctx = v.ctx;
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, wpx, hpx);
  const s = Math.min(wpx, hpx);
  const ox = (wpx - s) / 2, oy = (hpx - s) / 2;
  ctx.drawImage(v.buf, ox, oy, s, s);

  const px2view = s / H;
  const cx = ox + s / 2, cy = oy + s / 2;
  // Airy 第一暗环
  if (p.airyPx > 0) {
    ctx.beginPath();
    ctx.arc(cx, cy, p.airyPx * px2view, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(215, 219, 226, 0.7)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  // 中心十字
  const a = Math.max(4, s * 0.03);
  ctx.beginPath();
  ctx.moveTo(cx - a, cy); ctx.lineTo(cx + a, cy);
  ctx.moveTo(cx, cy - a); ctx.lineTo(cx, cy + a);
  ctx.strokeStyle = "#d7dbe2";
  ctx.stroke();
}
