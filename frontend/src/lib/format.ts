export const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`;
export const num = (x: number, d = 3) => x.toFixed(d);
export const signed = (x: number, d = 2) => `${x >= 0 ? "+" : ""}${x.toFixed(d)}`;
export const clock = (t: number) => {
  const d = new Date(t);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}.${String(d.getMilliseconds()).padStart(3, "0")}`;
};
export const cx = (...xs: (string | false | null | undefined | 0)[]) =>
  xs.filter((x): x is string => typeof x === "string" && x.length > 0).join(" ");
