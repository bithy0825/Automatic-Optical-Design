import { expect, test } from "bun:test";
import { airyDiameterMm, centroid, fmtUm, geoRadius, rmsRadius } from "./annotate";

test("centroid 中心与 NaN 跳过", () => {
  expect(centroid([1, 1, -1, 1, -1, -1, 1, -1])).toEqual([0, 0]);
  const [x, y] = centroid([NaN, 0, 2, 2, 2, 4]);
  expect(x).toBeCloseTo(2);
  expect(y).toBeCloseTo(3);
});

test("rmsRadius / geoRadius", () => {
  const pts = [3, 0, 0, 4]; // 相对原点 r=3, r=4
  expect(rmsRadius(pts, [0, 0])).toBeCloseTo(Math.sqrt((9 + 16) / 2));
  expect(geoRadius(pts, [0, 0])).toBeCloseTo(4);
  expect(rmsRadius(pts, [3, 0])).toBeCloseTo(Math.sqrt((0 + 25) / 2));
});

test("airyDiameterMm", () => {
  expect(airyDiameterMm(550, 10)).toBeCloseTo(2.44 * 550e-6 * 10, 6); // ≈ 0.0134 mm
});

test("fmtUm", () => {
  expect(fmtUm(0.01234)).toBe("12.3 μm");
  expect(fmtUm(1.2)).toBe("1200.0 μm");
});
