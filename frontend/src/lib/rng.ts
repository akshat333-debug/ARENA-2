/** Deterministic RNG (mulberry32). Seeded runs must reproduce exactly — the
 *  same guarantee the Python side gives (arena/config.py `seed`). */
export function makeRng(seed: number) {
  let a = seed >>> 0;
  const next = () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    next,
    int: (n: number) => Math.floor(next() * n),
    pick: <T,>(xs: readonly T[]) => xs[Math.floor(next() * xs.length)],
    shuffle: <T,>(xs: readonly T[]) => {
      const a2 = [...xs];
      for (let i = a2.length - 1; i > 0; i--) {
        const j = Math.floor(next() * (i + 1));
        [a2[i], a2[j]] = [a2[j], a2[i]];
      }
      return a2;
    },
    chance: (p: number) => next() < p,
  };
}
export type Rng = ReturnType<typeof makeRng>;
