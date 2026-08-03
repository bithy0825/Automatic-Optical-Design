/** AODV 二进制帧解码 —— visualization/protocol.py 的 TS 镜像(改动须同步)。 */

export interface Tensor {
  data: Float32Array | Uint8Array;
  shape: number[];
}

export interface Packet<M> {
  meta: M;
  sections: Record<string, Tensor>;
}

const MAGIC = 0x56444f41; // "AODV" 小端
const VERSION = 1;

export function decode<M>(buf: ArrayBuffer): Packet<M> {
  const dv = new DataView(buf);
  if (buf.byteLength < 9 || dv.getUint32(0, true) !== MAGIC) {
    throw new Error("decode: bad magic");
  }
  if (dv.getUint8(4) !== VERSION) {
    throw new Error(`decode: unsupported version ${dv.getUint8(4)}`);
  }
  const hlen = dv.getUint32(5, true);
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buf, 9, hlen)),
  ) as {
    meta: M;
    sections: Record<string, { offset: number; shape: number[]; dtype: "f32" | "u8" }>;
  };
  const base = 9 + hlen;
  const sections: Record<string, Tensor> = {};
  for (const [name, d] of Object.entries(header.sections)) {
    const count = d.shape.reduce((a, b) => a * b, 1);
    sections[name] = {
      data:
        d.dtype === "f32"
          ? new Float32Array(buf, base + d.offset, count)
          : new Uint8Array(buf, base + d.offset, count),
      shape: d.shape,
    };
  }
  return { meta: header.meta, sections };
}

async function check(r: Response): Promise<Response> {
  if (!r.ok) {
    let msg = `${r.status}`;
    try {
      msg = (await r.json()).error ?? msg;
    } catch {
      /* 非 JSON 错误体,保留状态码 */
    }
    throw new Error(msg);
  }
  return r;
}

export async function fetchJson<T>(url: string): Promise<T> {
  return (await check(await fetch(url))).json();
}

export async function fetchPacket<M>(url: string): Promise<Packet<M>> {
  return decode<M>(await (await check(await fetch(url))).arrayBuffer());
}

/** /api/system */
export interface SystemMeta {
  population: number;
  epd: number;
  fields_deg: [number, number][];
  wavelengths_nm: number[];
  surfaces: { index: number; label: string; kind: string }[];
  target: { id: string; fov: number | number[] | number[][]; F: number; effl: number } | null;
}

/** /api/layout 包 meta */
export interface LayoutMeta {
  labels: string[];
  kinds: string[];
  regions: string[]; // S+1 段:regions[0] 为首面上游,regions[j] 为面 j-1 下游
  effl: number; // 最小二乘估计焦距 mm(与 effl_loss 同一估计量)
  total_length: number; // 首面顶点 → 传感器总长 mm
  fields_deg: [number, number][];
  wavelengths_nm: number[];
}

/** /api/spot 包 meta */
export interface SpotMeta {
  chief_index: number; // 主光线在 N 维中的索引(恒 0)
  fields_deg: [number, number][];
  wavelengths_nm: number[];
}
