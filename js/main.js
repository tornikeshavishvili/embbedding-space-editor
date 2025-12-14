import { state, uid } from "./state.js";
import { zeros, normalizeInPlace, parseVector, vecNorm } from "./math.js";
import { getEls } from "./dom.js";
import { ensurePCAComputed, recomputePCAExplicit, meanVector, pcaResidualForItem, pcaReconstruct } from "./pca.js";
import { draw, hitTestPoint, getCanvasSize, screenToWorld } from "./plot.js";
import { rerenderList, syncSelectedUI, updateNeighbors, getSelected, initNeighborBarSlider, appendNewItemToFrozenOrder } from "./ui.js";

const els = getEls();

function setEditorMode(mode) {
  // mode: "add" | "edit"
  if (mode === "edit") {
    els.addBtn.textContent = "Update";
    els.addBtn.dataset.mode = "edit";
  } else {
    els.addBtn.textContent = "Add";
    els.addBtn.dataset.mode = "add";
    els.textInput.value = "";
    els.tokenInput.value = "";
    // keep type selection as user last used
  }
}

function loadSelectedIntoEditor() {
  const it = getSelected();
  if (!it) { setEditorMode("add"); return; }
  els.textInput.value = it.text;
  els.typeInput.value = it.type;
  els.tokenInput.value = it.token || "";
  setEditorMode("edit");
}


function onSelect(id) {
  state.selectedId = id;
  loadSelectedIntoEditor();
  if (!state.selectedId) setEditorMode("add");
  syncSelectedUI(els);
  loadSelectedIntoEditor();
  if (!state.selectedId) setEditorMode("add");
  draw(els.canvas);
}

function onDelete(id) {
  state.items = state.items.filter(x => x.id !== id);
  if (state.selectedId === id) state.selectedId = null;
  setEditorMode("add");

  recomputePCAExplicit();
  rerenderList(els, onSelect, onDelete);
  syncSelectedUI(els);
  draw(els.canvas);
}

function setTargetCosine(neighborId, targetCos) {
  const sel = getSelected();
  if (!sel) return;
  const other = state.items.find(x => x.id === neighborId);
  if (!other) return;

  let t = Math.max(-0.999999, Math.min(0.999999, targetCos));

  normalizeInPlace(sel.vector);
  normalizeInPlace(other.vector);

  const dot = sel.vector.reduce((s, x, i) => s + x * other.vector[i], 0);
  let w = sel.vector.map((x, i) => x - dot * other.vector[i]);
  let wn = Math.sqrt(w.reduce((s, x) => s + x * x, 0));

  if (wn < 1e-10) {
    w = other.vector.map((_, i) => (i === 0 ? 1 : 0));
    const d2 = w.reduce((s, x, i) => s + x * other.vector[i], 0);
    w = w.map((x, i) => x - d2 * other.vector[i]);
    wn = Math.sqrt(w.reduce((s, x) => s + x * x, 0));
    if (wn < 1e-10) return;
  }
  w = w.map(x => x / wn);

  const a = t;
  const b = Math.sqrt(Math.max(0, 1 - t * t));

  sel.vector = other.vector.map((x, i) => a * x + b * w[i]);
  normalizeInPlace(sel.vector);

  state.pcaLocked = true;

  const idx = state.items.findIndex(x => x.id === sel.id);
  if (idx >= 0 && state.pca.W && state.pca.mean?.length) {
    const centered = sel.vector.map((x, i) => x - state.pca.mean[i]);
    const w1 = state.pca.W[0];
    const w2 = state.pca.W[1];
    const x2 = centered.reduce((s, v, i) => s + v * w1[i], 0);
    const y2 = centered.reduce((s, v, i) => s + v * w2[i], 0);
    state.pca.pts2[idx] = [x2, y2];
  }
}


function uniqueId(existingIds) {
  let id;
  do { id = uid(); } while (existingIds.has(id));
  existingIds.add(id);
  return id;
}

function coerceVector(vec, dim) {
  const v = (Array.isArray(vec) ? vec.map(Number) : zeros(dim)).slice(0, dim);
  while (v.length < dim) v.push(0);
  return v;
}

function mergePackIntoState(pack, sourceLabel = "") {
  const items = pack?.items;
  const dim = pack?.metadata?.dim;

  if (!Array.isArray(items)) throw new Error("Bad pack: items missing");
  if (!Number.isFinite(dim)) throw new Error("Bad pack: metadata.dim missing");

  // If empty state, adopt pack dim
  if (state.items.length === 0) {
    state.dim = Math.max(2, Math.min(256, dim));
    els.dimInput.value = String(state.dim);
  }

  // If pack dim differs, we adapt vectors to current dim (no rejection)
  const targetDim = state.dim;

  const existingIds = new Set(state.items.map(x => x.id));
  const addedIds = [];

  for (const x of items) {
    const it = {
      id: (x.id && !existingIds.has(x.id)) ? (existingIds.add(x.id), x.id) : uniqueId(existingIds),
      text: String(x.text || ""),
      type: String(x.type || "word"),
      token: String(x.token || ""),
      vector: coerceVector(x.vector, targetDim),
      _source: sourceLabel || ""
    };

    if (vecNorm(it.vector) > 1e-12) normalizeInPlace(it.vector);
    state.items.push(it);
    addedIds.push(it.id);
  }

  // PCA: if basis exists, project new points without moving old ones
  if (state.pca.W && state.pca.mean?.length) {
    state.pcaLocked = true;
    const w1 = state.pca.W[0];
    const w2 = state.pca.W[1];
    for (let k = 0; k < addedIds.length; k++) {
      const it = state.items[state.items.length - addedIds.length + k];
      const centered = it.vector.map((v, i) => v - state.pca.mean[i]);
      const x2 = centered.reduce((s, v, i) => s + v * w1[i], 0);
      const y2 = centered.reduce((s, v, i) => s + v * w2[i], 0);
      state.pca.pts2.push([x2, y2]);
    }
  } else {
    recomputePCAExplicit();
  }

  return addedIds.length;
}



/* Buttons */
els.addBtn.onclick = () => {
  const text = els.textInput.value.trim();
  if (!text) return;

  const mode = els.addBtn.dataset.mode || "add";

  if (mode === "edit" && state.selectedId) {
    const it = getSelected();
    if (!it) { setEditorMode("add"); return; }

    it.text = text;
    it.type = els.typeInput.value;
    it.token = els.tokenInput.value.trim() || "";

    rerenderList(els, onSelect, onDelete);
    syncSelectedUI(els);
    draw(els.canvas);
    return;
  }

  // Add new item WITHOUT changing current selection (so it appears as a new neighbor)
  const prevSelectedId = state.selectedId;

  const newItem = {
    id: uid(),
    text,
    type: els.typeInput.value,
    token: els.tokenInput.value.trim() || "",
    vector: Array(state.dim).fill(0).map(() => (Math.random() * 2 - 1))
  };
  normalizeInPlace(newItem.vector);

  // Add at the END to keep existing PCA point indices aligned
  state.items.push(newItem);

  // Keep selection if it existed; otherwise select the new item
  if (!prevSelectedId) state.selectedId = newItem.id;

  // Keep neighbor list stable: append new id (no reorder)
  if (prevSelectedId) appendNewItemToFrozenOrder(newItem.id);

  // Make it appear in PCA view immediately:
  // If PCA basis exists, just project into current basis (no rearrange).
  // Otherwise recompute once.
  if (state.pca.W && state.pca.mean?.length) {
    state.pcaLocked = true; // do not recompute PCA automatically
    const centered = newItem.vector.map((x, i) => x - state.pca.mean[i]);
    const w1 = state.pca.W[0];
    const w2 = state.pca.W[1];
    const x2 = centered.reduce((s, v, i) => s + v * w1[i], 0);
    const y2 = centered.reduce((s, v, i) => s + v * w2[i], 0);
    state.pca.pts2.push([x2, y2]);
  } else {
    recomputePCAExplicit();
  }

  // Update UI
  rerenderList(els, onSelect, onDelete);
  syncSelectedUI(els);
  updateNeighbors(els);
  draw(els.canvas);

  // Reset editor for next add (but keep current type)
  if (!prevSelectedId) loadSelectedIntoEditor();
  if (!state.selectedId) setEditorMode("add");
  else setEditorMode("add");
};

els.newBtn.onclick = () => {
  // Switch editor to Add mode without changing PCA/selection unless user wants a blank add
  state.selectedId = null;
  setEditorMode("add");
  syncSelectedUI(els);
  rerenderList(els, onSelect, onDelete);
  draw(els.canvas);
};


els.resizeBtn.onclick = () => {
  const d = Math.max(2, Math.min(256, parseInt(els.dimInput.value || "8", 10)));
  state.dim = d;

  for (const it of state.items) {
    it.vector = it.vector.slice(0, d);
    while (it.vector.length < d) it.vector.push(0);
    if (vecNorm(it.vector) > 1e-12) normalizeInPlace(it.vector);
  }

  recomputePCAExplicit();
  rerenderList(els, onSelect, onDelete);
  syncSelectedUI(els);
  draw(els.canvas);
};

els.searchInput.oninput = () => rerenderList(els, onSelect, onDelete);

els.randomBtn.onclick = () => {
  const it = getSelected(); if (!it) return;
  it.vector = Array(state.dim).fill(0).map(() => (Math.random() * 2 - 1));
  normalizeInPlace(it.vector);
  recomputePCAExplicit();
  syncSelectedUI(els);
  draw(els.canvas);
};

els.normalizeBtn.onclick = () => {
  const it = getSelected(); if (!it) return;
  normalizeInPlace(it.vector);
  recomputePCAExplicit();
  syncSelectedUI(els);
  draw(els.canvas);
};

els.saveVecBtn.onclick = () => {
  const it = getSelected(); if (!it) return;
  const v = parseVector(els.vectorArea.value, state.dim);
  if (!v) { alert("Vector parse error."); return; }
  it.vector = v;
  if (vecNorm(it.vector) > 1e-12) normalizeInPlace(it.vector);
  recomputePCAExplicit();
  syncSelectedUI(els);
  draw(els.canvas);
};

els.exportBtn.onclick = () => {
  const pack = {
    metadata: { version: 1, createdAt: new Date().toISOString(), dim: state.dim, method: "manual-gui" },
    items: state.items.map(it => ({ id: it.id, text: it.text, type: it.type, token: it.token, vector: it.vector }))
  };
  els.jsonArea.value = JSON.stringify(pack, null, 2);
};

els.importBtn.onclick = () => {
  const t = els.jsonArea.value.trim();
  if (!t) return;

  let pack;
  try { pack = JSON.parse(t); } catch { alert("Invalid JSON"); return; }

  try {
    const added = mergePackIntoState(pack, "textarea");
    if (!state.selectedId && state.items.length) state.selectedId = state.items[0].id;
    rerenderList(els, onSelect, onDelete);
    syncSelectedUI(els);
    updateNeighbors(els);
    draw(els.canvas);
    alert(`Merged ${added} items.`);
  } catch (e) {
    alert(e?.message || "Import failed");
  }
};

els.importFiles.onchange = async () => {
  const files = Array.from(els.importFiles.files || []);
  if (files.length === 0) return;

  let total = 0;
  for (const file of files) {
    try {
      const text = await file.text();
      const pack = JSON.parse(text);
      total += mergePackIntoState(pack, file.name);
    } catch (e) {
      alert(`Failed to import ${file.name}: ${e?.message || "error"}`);
    }
  }

  // reset file input so user can import same files again later
  els.importFiles.value = "";

  if (!state.selectedId && state.items.length) state.selectedId = state.items[0].id;

  rerenderList(els, onSelect, onDelete);
  syncSelectedUI(els);
  updateNeighbors(els);
  draw(els.canvas);

  alert(`Merged ${total} items from ${files.length} file(s).`);
};


els.pcaBtn.onclick = () => { recomputePCAExplicit(); draw(els.canvas); };

els.centerBtn.onclick = () => {
  if (state.items.length === 0) return;
  const mu = meanVector(state.items.map(it => it.vector), state.dim);
  for (const it of state.items) {
    for (let i = 0; i < state.dim; i++) it.vector[i] -= mu[i];
    if (vecNorm(it.vector) > 1e-12) normalizeInPlace(it.vector);
  }
  recomputePCAExplicit();
  syncSelectedUI(els);
  draw(els.canvas);
};

els.resetViewBtn.onclick = () => {
  state.view.scale = 180;
  state.view.ox = 0;
  state.view.oy = 0;
  draw(els.canvas);
};

els.seedDemoBtn.onclick = () => {
  const seed = [
    { text: "block open", type: "phrase", token: "{" },
    { text: "block close", type: "phrase", token: "}" },
    { text: "repeat", type: "word", token: "for" },
    { text: "times", type: "word", token: "" },
    { text: "print", type: "word", token: "console.log" },
    { text: "set", type: "word", token: "=" },
    { text: "to", type: "word", token: "" },
    { text: "counter", type: "word", token: "identifier" },
    { text: "if", type: "word", token: "if" },
    { text: "else", type: "word", token: "else" },
    { text: "(", type: "token", token: "(" },
    { text: ")", type: "token", token: ")" },
  ];

  for (const s of seed) {
    const it = {
      id: uid(),
      text: s.text,
      type: s.type,
      token: s.token,
      vector: Array(state.dim).fill(0).map(() => (Math.random() * 2 - 1))
    };
    normalizeInPlace(it.vector);
    state.items.push(it);
  }

  state.selectedId = state.items[0]?.id || null;
  recomputePCAExplicit();
  rerenderList(els, onSelect, onDelete);
  syncSelectedUI(els);
  draw(els.canvas);
};

els.clearBtn.onclick = () => {
  state.items = [];
  state.selectedId = null;
  state.pca = { pts2: [], mean: [], W: null, lambdas: [0, 0], hash: "" };
  state.pcaLocked = false;
  rerenderList(els, onSelect, onDelete);
  syncSelectedUI(els);
  draw(els.canvas);
};

/* Mouse interactions (point dragging / panning) */
els.canvas.addEventListener("mousedown", (e) => {
  const rect = els.canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  if (!state.pcaLocked || !state.pca.W) ensurePCAComputed();

  const id = hitTestPoint(els.canvas, mx, my);

  state.dragging.on = true;
  state.dragging.id = id;
  state.dragging.dx = mx;
  state.dragging.dy = my;
  state.dragging.residual = null;
  state.dragging.index = -1;

  if (id) {
    state.selectedId = id;
    syncSelectedUI(els);
    rerenderList(els, onSelect, onDelete);

    const idx = state.items.findIndex(x => x.id === id);
    state.dragging.index = idx;
    state.dragging.residual = pcaResidualForItem(idx);
  }

  draw(els.canvas);
});

window.addEventListener("mouseup", () => {
  if (state.dragging.id) state.pcaLocked = true;
  state.dragging.on = false;
  state.dragging.id = null;
  state.dragging.residual = null;
  state.dragging.index = -1;
});

window.addEventListener("mousemove", (e) => {
  if (!state.dragging.on) return;

  const rect = els.canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const { w, h } = getCanvasSize(els.canvas);

  if (state.dragging.id) {
    const idx = state.dragging.index;
    const it = getSelected();
    if (!it || idx < 0) return;

    const [xNew, yNew] = screenToWorld(mx, my, w, h);

    const mean = state.pca.mean;
    const W1 = state.pca.W?.[0];
    const W2 = state.pca.W?.[1];
    const r = state.dragging.residual;
    if (!mean || !W1 || !W2 || !r) return;

    it.vector = pcaReconstruct(mean, W1, W2, xNew, yNew, r);
    if (vecNorm(it.vector) > 1e-12) normalizeInPlace(it.vector);

    if (state.pca.pts2[idx]) state.pca.pts2[idx] = [xNew, yNew];

    // Live cosine bar updates
    updateNeighbors(els);
    draw(els.canvas);
  } else {
    const dx = mx - state.dragging.dx;
    const dy = my - state.dragging.dy;
    state.view.ox += dx;
    state.view.oy += dy;
    state.dragging.dx = mx;
    state.dragging.dy = my;
    draw(els.canvas);
  }
});

els.canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const delta = Math.sign(e.deltaY);
  const factor = (delta > 0) ? 0.9 : 1.1;
  state.view.scale = Math.max(20, Math.min(2000, state.view.scale * factor));
  draw(els.canvas);
}, { passive: false });

function boot() {
  state.dim = parseInt(els.dimInput.value, 10) || 8;
  rerenderList(els, onSelect, onDelete);
  syncSelectedUI(els);
  draw(els.canvas);

  // Activate bar-as-slider behavior
  initNeighborBarSlider(
    els,
    (neiId, targetCos) => {
      setTargetCosine(neiId, targetCos);
      els.vectorArea.value = getSelected()?.vector.map(x => (Math.round(x * 1000) / 1000)).join(", ") || "";
    },
    () => {
      updateNeighbors(els);
      draw(els.canvas);
    }
  );
}
boot();
