import { expect, test } from "bun:test";
import { fieldColor, materialColor, wavelengthColor } from "./palette";

function rgb(css: string): [number, number, number] {
  const m = css.match(/rgb\((\d+),(\d+),(\d+)\)/)!;
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}

test("wavelengthColor 光谱顺序", () => {
  const [r1, g1, b1] = rgb(wavelengthColor(486));
  expect(b1).toBeGreaterThan(r1);            // 蓝紫段
  const [r2, g2, b2] = rgb(wavelengthColor(530));
  expect(g2).toBeGreaterThan(b2);
  expect(g2).toBeGreaterThan(0);             // 绿段
  const [r3, g3, b3] = rgb(wavelengthColor(656));
  expect(r3).toBeGreaterThan(g3);
  expect(b3).toBe(0);                        // 红段无蓝
});

test("wavelengthColor 超界钳位不抛错", () => {
  expect(() => wavelengthColor(200)).not.toThrow();
  expect(() => wavelengthColor(1200)).not.toThrow();
});

test("fieldColor 循环稳定", () => {
  expect(fieldColor(0)).toBe(fieldColor(0));
  expect(fieldColor(0)).not.toBe(fieldColor(1));
  expect(fieldColor(8)).toBe(fieldColor(0));
});

test("materialColor 同名同色", () => {
  expect(materialColor("N-BK7")).toBe(materialColor("N-BK7"));
  expect(materialColor("N-BK7")).not.toBe(materialColor("N-SF6"));
  expect(materialColor("N-BK7")).toMatch(/^hsl\(/);
});
