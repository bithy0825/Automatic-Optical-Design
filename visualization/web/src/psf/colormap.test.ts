import { describe, expect, test } from "bun:test";
import { inferno } from "./colormap";

describe("inferno", () => {
  test("端点", () => {
    expect(inferno(0)).toEqual([0, 0, 4]);
    expect(inferno(1)).toEqual([252, 255, 164]);
  });
  test("越界钳制", () => {
    expect(inferno(-1)).toEqual(inferno(0));
    expect(inferno(2)).toEqual(inferno(1));
  });
  test("中点落在通道合法范围", () => {
    for (const t of [0.13, 0.37, 0.5, 0.62, 0.88]) {
      const [r, g, b] = inferno(t);
      for (const c of [r, g, b]) {
        expect(c).toBeGreaterThanOrEqual(0);
        expect(c).toBeLessThanOrEqual(255);
        expect(Number.isInteger(c)).toBe(true);
      }
    }
  });
  test("亮度大致单调(inferno 感知特性)", () => {
    let prev = -1;
    for (let i = 0; i <= 16; i++) {
      const [r, g, b] = inferno(i / 16);
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      expect(lum).toBeGreaterThanOrEqual(prev);
      prev = lum;
    }
  });
});
