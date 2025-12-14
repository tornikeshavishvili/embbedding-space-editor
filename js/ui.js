import { state } from "./state.js";
import { cosine, escapeHtml } from "./math.js";

let lastScores = new Map();
let frozenOrder = [];
let frozenForSelectedId = null;

export function appendNewItemToFrozenOrder(newId) {
  const it = getSelected();
  if (!it) return;
  // If neighbor order frozen for this selected item, append new item at end (no reordering)
  if (frozenForSelectedId === it.id) {
    if (newId && newId !== it.id && !frozenOrder.includes(newId)) {
      frozenOrder.push(newId);
    }
  }
}

function getById(id) {
  return state.items.find(x => x.id === id) || null;
}

export function getSelected() {
  return getById(state.selectedId);
}

export function freezeNeighborsOrder() {
  const it = getSelected();
  if (!it) {
    frozenOrder = [];
    frozenForSelectedId = null;
    lastScores.clear();
    return;
  }
  if (frozenForSelectedId === it.id && frozenOrder.length) return;

  const scored = state.items
    .filter(x => x.id !== it.id)
    .map(x => ({ id: x.id, score: cosine(it.vector, x.vector) }))
    .sort((a, b) => b.score - a.score);

  frozenOrder = scored.map(s => s.id);
  frozenForSelectedId = it.id;
  lastScores.clear();
}

export function rerenderList(els, onSelect, onDelete) {
  const q = els.searchInput.value.trim().toLowerCase();
  els.list.innerHTML = "";

  const items = state.items.filter(it => {
    if (!q) return true;
    return it.text.toLowerCase().includes(q) || (it.token || "").toLowerCase().includes(q);
  });

  for (const it of items) {
    const row = document.createElement("div");
    row.className = "item" + (it.id === state.selectedId ? " active" : "");
    row.onclick = () => onSelect(it.id);

    const left = document.createElement("div");
    left.innerHTML = `
      <div><strong>${escapeHtml(it.text)}</strong></div>
      <div class="muted">${escapeHtml(it.token || "")}</div>
      <div class="small"><span class="pill">${escapeHtml(it.type)}</span></div>
    `;

    const right = document.createElement("div");
    right.className = "actions";

    const del = document.createElement("button");
    del.textContent = "Delete";
    del.onclick = (e) => { e.stopPropagation(); onDelete(it.id); };
    right.appendChild(del);

    row.appendChild(left);
    row.appendChild(right);
    els.list.appendChild(row);
  }
}

export function syncSelectedUI(els) {
  const it = getSelected();
  if (!it) {
    els.selectedMeta.textContent = "No item selected.";
    els.vectorArea.value = "";
    els.neighbors.innerHTML = "";
    frozenOrder = [];
    frozenForSelectedId = null;
    lastScores.clear();
    return;
  }

  freezeNeighborsOrder();

  els.selectedMeta.innerHTML = `
    <div><strong>${escapeHtml(it.text)}</strong> <span class="pill">${escapeHtml(it.type)}</span></div>
    <div class="muted">token/op: <span class="mono">${escapeHtml(it.token || "(none)")}</span></div>
    <div class="muted">id: <span class="mono">${escapeHtml(it.id)}</span></div>
  `;

  els.vectorArea.value = it.vector.map(x => (Math.round(x * 1000) / 1000)).join(", ");
  updateNeighbors(els);
}

export function updateNeighbors(els, opts = {}) {
  const it = getSelected();
  if (!it) { els.neighbors.innerHTML = ""; return; }

  const alive = new Set(state.items.map(x => x.id));
  frozenOrder = frozenOrder.filter(id => alive.has(id) && id !== it.id);

  const limit = Number.isFinite(opts.limit) ? opts.limit : 12;
  const draggingNeighborId = opts.draggingNeighborId || null;

  const rows = [];
  for (const id of frozenOrder.slice(0, limit)) {
    const other = getById(id);
    if (!other) continue;

    const score = cosine(it.vector, other.vector);

    const prev = lastScores.get(id);
    const delta = prev == null ? 0 : score - prev;
    lastScores.set(id, score);

    const pct = Math.max(0, Math.min(1, (score + 1) / 2));
    const barWidth = Math.round(pct * 100);

    let arrow = "";
    let cls = "same";
    if (delta > 0.002) { arrow = "↑"; cls = "up"; }
    else if (delta < -0.002) { arrow = "↓"; cls = "down"; }

    const dragCls = (draggingNeighborId === id) ? " dragging" : "";

    rows.push(`
      <div class="sim-row ${cls}${dragCls}" data-nei-id="${escapeHtml(id)}">
        <div class="sim-label" title="${escapeHtml(other.text)}">
          <span class="sim-arrow">${arrow}</span>${escapeHtml(other.text)}
        </div>
        <div class="sim-bar" data-nei-id="${escapeHtml(id)}" style="--knob-x:${barWidth}%">
          <div class="sim-bar-fill" style="width:${barWidth}%"></div>
        </div>
        <div class="sim-score">${score.toFixed(3)}</div>
      </div>
    `);
  }

  els.neighbors.innerHTML = rows.join("");
}

export function initNeighborBarSlider(els, onSetTargetCosine, onAfterUpdate) {
  let dragging = false;
  let activeId = null;

  function clamp01(x) { return Math.max(0, Math.min(1, x)); }

  function barToTargetCos(barEl, clientX) {
    const rect = barEl.getBoundingClientRect();
    const t = clamp01((clientX - rect.left) / Math.max(1, rect.width));
    return (t * 2) - 1;
  }

  function findBarEl(target) {
    if (!target) return null;
    return target.closest?.(".sim-bar") || null;
  }

  els.neighbors.addEventListener("mousedown", (e) => {
    const barEl = findBarEl(e.target);
    if (!barEl) return;

    const id = barEl.dataset.neiId;
    if (!id) return;

    dragging = true;
    activeId = id;

    const targetCos = barToTargetCos(barEl, e.clientX);
    onSetTargetCosine(id, targetCos);

    updateNeighbors(els, { draggingNeighborId: activeId });
    if (typeof onAfterUpdate === "function") onAfterUpdate();

    e.preventDefault();
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragging || !activeId) return;

    // Find current bar element by dataset (no CSS.escape needed)
    let barEl = null;
    const bars = els.neighbors.querySelectorAll(".sim-bar");
    for (const b of bars) {
      if (b.dataset && b.dataset.neiId === activeId) { barEl = b; break; }
    }
    if (!barEl) return;

    const targetCos = barToTargetCos(barEl, e.clientX);
    onSetTargetCosine(activeId, targetCos);

    updateNeighbors(els, { draggingNeighborId: activeId });
    if (typeof onAfterUpdate === "function") onAfterUpdate();
  });

  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    activeId = null;

    updateNeighbors(els);
    if (typeof onAfterUpdate === "function") onAfterUpdate();
  });
}
