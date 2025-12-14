import { state } from "./state.js";
import { ensurePCAComputed } from "./pca.js";

export function getCanvasSize(canvas) {
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(300, Math.floor(rect.width));
  const h = Math.max(200, Math.floor(rect.height));
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  return { w, h };
}

export function worldToScreen(x, y, w, h) {
  const { scale, ox, oy } = state.view;
  const sx = (w / 2) + (x * scale) + ox;
  const sy = (h / 2) - (y * scale) + oy;
  return [sx, sy];
}

export function screenToWorld(sx, sy, w, h) {
  const { scale, ox, oy } = state.view;
  const x = (sx - (w / 2) - ox) / scale;
  const y = - (sy - (h / 2) - oy) / scale;
  return [x, y];
}

export function draw(canvas) {
  const { w, h } = getCanvasSize(canvas);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);

  ctx.globalAlpha = 1;
  ctx.strokeStyle = "#132033";
  ctx.lineWidth = 1;
  const step = 60;
  for (let x = 0; x <= w; x += step) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
  for (let y = 0; y <= h; y += step) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

  ctx.strokeStyle = "#1f2a37";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(0, h / 2 + state.view.oy); ctx.lineTo(w, h / 2 + state.view.oy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(w / 2 + state.view.ox, 0); ctx.lineTo(w / 2 + state.view.ox, h); ctx.stroke();

  if (!(state.dragging.on && state.dragging.id)) {
    ensurePCAComputed();
  } else {
    if (!state.pca.W) ensurePCAComputed();
  }

  const pts = state.pca.pts2;

  for (let i = 0; i < state.items.length; i++) {
    const it = state.items[i];
    const p = pts[i] || [0, 0];
    const [sx, sy] = worldToScreen(p[0], p[1], w, h);
    const r = (it.id === state.selectedId) ? 8 : 6;

    ctx.beginPath();
    ctx.fillStyle = (it.id === state.selectedId) ? "#3b82f6" : "#93c5fd";
    ctx.globalAlpha = 0.9;
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fill();

    ctx.globalAlpha = 0.9;
    ctx.fillStyle = "#e6edf3";
    ctx.font = "12px system-ui";
    ctx.fillText(it.text, sx + r + 6, sy + 4);
  }
  ctx.globalAlpha = 1;
}

export function hitTestPoint(canvas, mx, my) {
  const { w, h } = getCanvasSize(canvas);
  const pts = state.pca.pts2;

  for (let i = 0; i < state.items.length; i++) {
    const it = state.items[i];
    const p = pts[i] || [0, 0];
    const [sx, sy] = worldToScreen(p[0], p[1], w, h);
    const r = (it.id === state.selectedId) ? 10 : 9;
    const dx = mx - sx, dy = my - sy;
    if (dx * dx + dy * dy <= r * r) return it.id;
  }
  return null;
}
