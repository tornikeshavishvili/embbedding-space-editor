export const state = {
  dim: 8,
  items: [],
  selectedId: null,

  pca: { pts2: [], mean: [], W: null, lambdas: [0, 0], hash: "" },
  pcaLocked: false,

  dragging: { on: false, id: null, dx: 0, dy: 0, residual: null, index: -1 },
  view: { scale: 180, ox: 0, oy: 0 },
};

export function uid() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}
