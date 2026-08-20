/** inferno 风格感知均匀色图:t∈[0,1] → RGB(0–255 整数)。 */

const STOPS: [number, number, number][] = [
  [0, 0, 4],       // 黑紫
  [72, 15, 116],   // 紫
  [159, 42, 99],   // 品红
  [212, 72, 66],   // 橙红
  [245, 125, 21],  // 橙
  [252, 194, 97],  // 浅橙
  [252, 255, 164], // 白黄
];

export function inferno(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(x));
  const f = x - i;
  const a = STOPS[i], b = STOPS[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}
