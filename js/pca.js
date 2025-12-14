import { zeros, vecDot, vecSub, normalizeInPlace } from "./math.js";
import { state } from "./state.js";

export function vecMulAdd(out, v, k) { for (let i = 0; i < out.length; i++) out[i] += v[i] * k; return out; }

export function pcaReconstruct(mean, W1, W2, x, y, residual) {
  const v = mean.slice();
  vecMulAdd(v, W1, x);
  vecMulAdd(v, W2, y);
  for (let i = 0; i < v.length; i++) v[i] += residual[i];
  return v;
}

export function pcaResidualForItem(index) {
  const it = state.items[index];
  const mean = state.pca.mean;
  const W1 = state.pca.W?.[0];
  const W2 = state.pca.W?.[1];
  const xy = state.pca.pts2[index] || [0, 0];
  if (!it || !mean || !W1 || !W2) return zeros(state.dim);

  const approx = pcaReconstruct(mean, W1, W2, xy[0], xy[1], zeros(state.dim));
  return vecSub(it.vector, approx);
}

export function meanVector(X, dim) {
  const m = zeros(dim);
  for (const x of X) for (let i = 0; i < dim; i++) m[i] += x[i];
  for (let i = 0; i < dim; i++) m[i] /= Math.max(1, X.length);
  return m;
}

export function covMatrix(Xc) {
  const n = Xc.length;
  const d = Xc[0]?.length ?? 0;
  const C = Array(d).fill(0).map(() => Array(d).fill(0));
  if (n === 0) return C;

  for (const x of Xc) {
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += x[i] * x[j];
  }
  const k = 1 / Math.max(1, n - 1);
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] *= k;
  return C;
}

export function matVecMul(M, v) {
  const out = Array(M.length).fill(0);
  for (let i = 0; i < M.length; i++) {
    let s = 0;
    const row = M[i];
    for (let j = 0; j < row.length; j++) s += row[j] * v[j];
    out[i] = s;
  }
  return out;
}

export function outer(a, b) {
  const M = Array(a.length);
  for (let i = 0; i < a.length; i++) {
    const row = Array(b.length);
    for (let j = 0; j < b.length; j++) row[j] = a[i] * b[j];
    M[i] = row;
  }
  return M;
}

export function matSub(A, B) {
  const M = Array(A.length);
  for (let i = 0; i < A.length; i++) {
    const row = Array(A[i].length);
    for (let j = 0; j < A[i].length; j++) row[j] = A[i][j] - B[i][j];
    M[i] = row;
  }
  return M;
}

export function enforceStableSign(v) {
  let idx = 0;
  let best = Math.abs(v[0] ?? 0);
  for (let i = 1; i < v.length; i++) {
    const a = Math.abs(v[i]);
    if (a > best) { best = a; idx = i; }
  }
  if ((v[idx] ?? 0) < 0) for (let i = 0; i < v.length; i++) v[i] = -v[i];
  return v;
}

export function deterministicInit(d, prev) {
  if (prev && prev.length === d) return prev.slice();
  const v = zeros(d); v[0] = 1; return v;
}

export function powerIterationSymmetric(C, iters = 80, initVec = null) {
  const d = C.length;
  if (d === 0) return { v: [], lambda: 0 };

  let v = deterministicInit(d, initVec);
  normalizeInPlace(v);

  for (let t = 0; t < iters; t++) {
    const Cv = matVecMul(C, v);
    normalizeInPlace(Cv);
    v = Cv;
  }
  const Cv = matVecMul(C, v);
  const lambda = vecDot(v, Cv);
  enforceStableSign(v);
  return { v, lambda };
}

export function computePCA2D(items, dim, prevW = null) {
  const X = items.map(it => it.vector.slice());
  const mu = meanVector(X, dim);
  const Xc = X.map(x => x.map((v, i) => v - mu[i]));
  const C = covMatrix(Xc);

  const e1 = powerIterationSymmetric(C, 80, prevW?.[0] ?? null);

  const vvT = outer(e1.v, e1.v);
  const C2 = matSub(C, vvT.map(row => row.map(x => x * e1.lambda)));

  const e2 = powerIterationSymmetric(C2, 80, prevW?.[1] ?? null);

  enforceStableSign(e1.v);
  enforceStableSign(e2.v);

  const pts2 = Xc.map(x => [vecDot(x, e1.v), vecDot(x, e2.v)]);
  return { pts2, mean: mu, W: [e1.v, e2.v], lambdas: [e1.lambda, e2.lambda] };
}

export function dataHash() {
  const parts = [String(state.dim), String(state.items.length)];
  for (let i = 0; i < state.items.length; i++) {
    parts.push(state.items[i].id);
    const v = state.items[i].vector;
    for (let j = 0; j < Math.min(8, v.length); j++) parts.push(String(Math.round(v[j] * 1e6)));
  }
  return parts.join("|");
}

export function ensurePCAComputed() {
  if (state.items.length < 2) {
    state.pca = { pts2: [], mean: [], W: null, lambdas: [0, 0], hash: "" };
    return;
  }

  if (state.pcaLocked && state.pca.W) return;

  const h = dataHash();
  if (!state.pca.W || state.pca.hash !== h) {
    const p = computePCA2D(state.items, state.dim, state.pca.W);
    state.pca = { ...p, hash: h };
  }
}

export function recomputePCAExplicit() {
  if (state.items.length < 2) return;
  const p = computePCA2D(state.items, state.dim, state.pca.W);
  state.pca = { ...p, hash: dataHash() };
  state.pcaLocked = false;
}
