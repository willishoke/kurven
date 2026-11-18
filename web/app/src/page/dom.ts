// The page's one DOM helper: an element with attributes and children.

type Child = Node | string | null | undefined | false;

/** `<tag attrs...>children</tag>`. An attribute of `false` or `undefined` is
 * left off, `true` is present and empty. */
export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number | boolean | undefined> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === false) continue;
    el.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c);
  }
  return el;
}

/** A value for a number field: as few digits as say it. */
export function shown(x: number, digits = 4): string {
  if (!Number.isFinite(x)) return "";
  return String(Number(x.toPrecision(digits)));
}
