export function zeros(n) { return Array(n).fill(0); }

export function vecDot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
export function vecNorm(a) { return Math.sqrt(Math.max(1e-18, vecDot(a, a))); }

export function vecSub(a, b) { return a.map((x, i) => x - b[i]); }

export function normalizeInPlace(v) {
  const n = vecNorm(v);
  for (let i = 0; i < v.length; i++) v[i] /= n;
}

export function cosine(a, b) {
  return vecDot(a, b) / (vecNorm(a) * vecNorm(b));
}

export function parseVector(text, dim) {
  const parts = text.replace(/\n/g, " ")
    .split(/[,\s]+/g)
    .map(s => s.trim())
    .filter(Boolean)
    .map(Number);
  if (parts.some(x => Number.isNaN(x))) return null;
  const v = parts.slice(0, dim);
  while (v.length < dim) v.push(0);
  return v;
}

export function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
