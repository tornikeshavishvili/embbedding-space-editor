# Embedding Space GUI

A browser-based visual editor for building, inspecting, and manipulating embedding spaces with **live cosine similarity**, **stable PCA visualization**, and **interactive semantic tuning**.

This tool is designed for **manual embedding construction and experimentation**, not neural training.

---

## Key Features

### 1. Embedding Pack Editor
- Create embedding items (word / phrase / token / AST op)
- Edit existing items (text, type, token) without changing vectors
- Explicit **Add** vs **Update** modes
- Duplicate words are allowed and preserved

### 2. Stable PCA Visualization (2D)
- Deterministic PCA (no random flips or rotations)
- PCA basis stays fixed while editing
- Drag a point → only that point moves
- Other points never jump or rotate unexpectedly

### 3. Interactive Cosine Similarity Sliders
- Nearest neighbors shown as **bars**
- Each bar is also a **slider**
- Drag a bar to:
  - change cosine similarity to that neighbor
  - update the selected vector
  - move the selected point in PCA space
- Neighbor order is **frozen** (no reordering while editing)

### 4. Multi-Pack Import & Merge
- Import one or multiple embedding packs
- Packs are **merged**, not replaced
- Same words appearing multiple times are all kept
- ID collisions handled automatically
- New items are projected into the existing PCA basis (no rearrange)

### 5. Layout
- **Left panel**: item editor + item list
- **Right side**:
  - PCA canvas
  - Selected item inspector
  - Vector editor
  - Cosine neighbors
  - Import / export
- Layout is stable during interaction (no canvas resizing while dragging)

---

## File Format (Embedding Pack)

```json
{
  "metadata": {
    "version": 1,
    "createdAt": "2025-01-01T00:00:00Z",
    "dim": 8,
    "method": "manual-gui"
  },
  "items": [
    {
      "id": "abc123",
      "text": "repeat",
      "type": "word",
      "token": "for",
      "vector": [0.12, -0.4, 0.7, 0, 0, 0, 0, 0]
    }
  ]
}
