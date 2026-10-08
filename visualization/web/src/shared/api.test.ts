import { expect, test } from "bun:test";
import { decode } from "./api";

/** 测试用独立编码器(protocol.py pack 的镜像,含对齐填充)。 */
function encode(
  meta: unknown,
  sections: Record<string, { data: Float32Array | Uint8Array; shape: number[] }>,
): ArrayBuffer {
  const secs: Record<string, { offset: number; shape: number[]; dtype: string }> = {};
  let payload = new Uint8Array(0);
  for (const [name, s] of Object.entries(sections)) {
    const bytes = new Uint8Array(s.data.buffer, s.data.byteOffset, s.data.byteLength);
    secs[name] = {
      offset: payload.length,
      shape: s.shape,
      dtype: s.data instanceof Float32Array ? "f32" : "u8",
    };
    const pad = (4 - (bytes.length % 4)) % 4; // 段后填充至 4 字节边界
    const next = new Uint8Array(payload.length + bytes.length + pad);
    next.set(payload);
    next.set(bytes, payload.length);
    payload = next;
  }
  let hj = new TextEncoder().encode(JSON.stringify({ meta, sections: secs }));
  const hpad = (8 - ((9 + hj.length) % 8)) % 8; // header 空格填充至 base % 8 == 0
  if (hpad) {
    const padded = new Uint8Array(hj.length + hpad);
    padded.set(hj);
    padded.fill(0x20, hj.length);
    hj = padded;
  }
  const out = new ArrayBuffer(9 + hj.length + payload.length);
  const dv = new DataView(out);
  dv.setUint32(0, 0x56444f41, true); // "AODV"
  dv.setUint8(4, 1);
  dv.setUint32(5, hj.length, true);
  new Uint8Array(out, 9, hj.length).set(hj);
  new Uint8Array(out, 9 + hj.length).set(payload);
  return out;
}

test("decode roundtrip", () => {
  const buf = encode(
    { hello: "world" },
    {
      a: { data: new Float32Array([1.5, -2, 3.25, 4]), shape: [2, 2] },
      b: { data: new Uint8Array([0, 1, 1]), shape: [3] },
    },
  );
  const p = decode<{ hello: string }>(buf);
  expect(p.meta.hello).toBe("world");
  expect(p.sections.a.shape).toEqual([2, 2]);
  expect(Array.from(p.sections.a.data as Float32Array)).toEqual([1.5, -2, 3.25, 4]);
  expect(p.sections.b.data instanceof Uint8Array).toBe(true);
});

test("decode 对齐:奇数 u8 段在前时 f32 段仍可读", () => {
  const buf = encode(
    { k: "v" },
    {
      u: { data: new Uint8Array([7, 8, 9]), shape: [3] }, // 3 字节,奇数
      f: { data: new Float32Array([42.5]), shape: [1, 1] },
    },
  );
  const p = decode<{ k: string }>(buf);
  expect(p.meta.k).toBe("v");
  expect(Array.from(p.sections.u.data as Uint8Array)).toEqual([7, 8, 9]);
  expect(Array.from(p.sections.f.data as Float32Array)).toEqual([42.5]);
});

test("decode rejects bad magic", () => {
  expect(() => decode(new ArrayBuffer(16))).toThrow("magic");
});
