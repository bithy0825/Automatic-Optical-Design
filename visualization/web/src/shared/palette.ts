/** 配色:波长→光谱色、视场循环色、材料名哈希色。 */

/** 可见光谱分段近似(380–780nm,两端向黑衰减模拟人眼边缘敏感度)。 */
export function wavelengthColor(nm: number): string {
  const w = Math.min(780, Math.max(380, nm));
  const b = w < 490 ? 1 : w < 510 ? (510 - w) / 20 : 0;
  const g = w < 440 ? 0 : w < 490 ? (w - 440) / 50 : w < 580 ? 1 : w < 645 ? (645 - w) / 65 : 0;
  const r = w < 440 ? (440 - w) / 60 : w < 510 ? 0 : w < 580 ? (w - 510) / 70 : 1;
  const fade = w < 420 ? 0.3 + (0.7 * (w - 380)) / 40 : w > 700 ? 0.3 + (0.7 * (780 - w)) / 80 : 1;
  const c = (v: number) => Math.round(255 * Math.min(1, Math.max(0, v * fade)));
  return `rgb(${c(r)},${c(g)},${c(b)})`;
}

/** Zemax 风格视场配色(循环)。 */
const FIELD_COLORS = [
  "#4da3ff", "#ff5a4e", "#5fd35f", "#ffb13d",
  "#c77dff", "#3fd4d4", "#ff7ac8", "#b8e04a",
];

export function fieldColor(i: number): string {
  return FIELD_COLORS[i % FIELD_COLORS.length];
}

/** 材料名 → 稳定填充色(哈希到 HSL 色相)。 */
export function materialColor(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return `hsl(${h % 360}, 45%, 62%)`;
}
