(() => {
  "use strict";

  const el = {
    list: document.getElementById("rv-list"),
    queuePath: document.getElementById("rv-queue-path"),
    progress: document.getElementById("rv-progress"),
    empty: document.getElementById("rv-empty"),
    workspace: document.getElementById("rv-workspace"),
    name: document.getElementById("rv-name"),
    patient: document.getElementById("rv-patient"),
    image: document.getElementById("rv-image"),
    overlay: document.getElementById("rv-overlay"),
    status: document.getElementById("rv-status"),
    add: document.getElementById("rv-add"),
    auto: document.getElementById("rv-auto"),
    acceptAll: document.getElementById("rv-accept-all"),
    clear: document.getElementById("rv-clear"),
    prev: document.getElementById("rv-prev"),
    next: document.getElementById("rv-next"),
    save: document.getElementById("rv-save"),
    filterAll: document.getElementById("rv-filter-all"),
    filterTodo: document.getElementById("rv-filter-todo"),
    filterDone: document.getElementById("rv-filter-done"),
  };

  const state = {
    records: [],
    index: -1,
    boxes: [],
    active: -1,
    addMode: false,
    filter: "all",
    autoTop1: localStorage.getItem("rv-auto-top1") !== "off",
    drag: null,
  };

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  function setStatus(message, kind) {
    el.status.textContent = message || "";
    el.status.className = "rv-status" + (kind ? " is-" + kind : "");
  }

  function updateAutoLabel() {
    el.auto.textContent = state.autoTop1 ? "Auto top-1: aktif" : "Auto top-1: mati";
    el.auto.classList.toggle("is-active", state.autoTop1);
  }

  function visibleIndices() {
    return state.records
      .map((r, i) => ({ r, i }))
      .filter(({ r }) => {
        if (state.filter === "todo") return !r.reviewed;
        if (state.filter === "done") return r.reviewed;
        return true;
      })
      .map(({ i }) => i);
  }

  function renderList() {
    el.list.innerHTML = "";
    const visible = new Set(visibleIndices());
    state.records.forEach((record, i) => {
      if (!visible.has(i)) return;
      const li = document.createElement("li");
      if (i === state.index) li.classList.add("is-active");
      if (record.reviewed) li.classList.add("is-done");
      const boxCount = record.reviewed
        ? (record.accepted || []).length
        : (record.candidates || []).length;
      li.innerHTML = `<span class="dot"></span><span>${record.image}</span><span class="count">${boxCount}</span>`;
      li.addEventListener("click", () => select(i));
      el.list.appendChild(li);
    });
    const done = state.records.filter((r) => r.reviewed).length;
    el.progress.textContent = `${done} / ${state.records.length} ditinjau`;
  }

  function loadBoxes(record) {
    if (record.reviewed && record.accepted && record.accepted.length) {
      return record.accepted.map((a) => ({ bbox: a.bbox.slice(), confidence: a.confidence || 0, accepted: true }));
    }
    const boxes = (record.candidates || []).map((c) => ({ bbox: c.bbox.slice(), confidence: c.confidence, accepted: false }));
    if (state.autoTop1 && !record.reviewed && boxes.length) {
      let best = 0;
      boxes.forEach((box, i) => { if (box.confidence > boxes[best].confidence) best = i; });
      boxes[best].accepted = true;
    }
    return boxes;
  }

  function select(i) {
    if (i < 0 || i >= state.records.length) return;
    state.index = i;
    state.active = -1;
    state.addMode = false;
    el.add.classList.remove("is-active");
    const record = state.records[i];
    state.boxes = loadBoxes(record);
    el.name.textContent = record.image;
    el.patient.textContent = record.patient || "";
    el.workspace.hidden = false;
    el.empty.hidden = true;
    setStatus("");
    el.image.src = `/api/review/image/${encodeURIComponent(record.image)}`;
    renderList();
  }

  function renderBoxes() {
    el.overlay.innerHTML = "";
    const w = el.image.naturalWidth;
    const h = el.image.naturalHeight;
    if (!w || !h) return;
    state.boxes.forEach((box, i) => {
      const [x1, y1, x2, y2] = box.bbox;
      const div = document.createElement("div");
      div.className = "rv-box" + (box.accepted ? " is-accepted" : "") + (i === state.active ? " is-active" : "");
      div.style.left = (x1 / w * 100) + "%";
      div.style.top = (y1 / h * 100) + "%";
      div.style.width = ((x2 - x1) / w * 100) + "%";
      div.style.height = ((y2 - y1) / h * 100) + "%";
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = (box.confidence ? box.confidence.toFixed(2) + " " : "") + (box.accepted ? "✓" : "");
      div.appendChild(tag);
      for (const corner of ["nw", "ne", "sw", "se"]) {
        const handle = document.createElement("span");
        handle.className = "rv-handle " + corner + (box.accepted ? " is-accepted" : "");
        handle.dataset.corner = corner;
        div.appendChild(handle);
      }
      div.addEventListener("pointerdown", (event) => startDrag(event, i));
      el.overlay.appendChild(div);
    });
  }

  function pointerToImage(event) {
    const rect = el.image.getBoundingClientRect();
    const scaleX = el.image.naturalWidth / rect.width;
    const scaleY = el.image.naturalHeight / rect.height;
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  }

  function startDrag(event, index) {
    if (event.button !== 0) return;
    const corner = event.target.dataset ? event.target.dataset.corner : null;
    if (state.addMode && !corner) return;
    event.preventDefault();
    state.active = index;
    const start = pointerToImage(event);
    state.drag = {
      index,
      corner: corner || null,
      start,
      origin: state.boxes[index].bbox.slice(),
      moved: false,
    };
    el.overlay.setPointerCapture(event.pointerId);
    renderBoxes();
  }

  function onPointerMove(event) {
    if (!state.drag) return;
    const point = pointerToImage(event);
    const { origin, start, corner } = state.drag;
    const dx = point.x - start.x;
    const dy = point.y - start.y;
    if (Math.abs(dx) + Math.abs(dy) > 2) state.drag.moved = true;
    let [x1, y1, x2, y2] = origin;
    const w = el.image.naturalWidth;
    const h = el.image.naturalHeight;
    if (!corner) {
      x1 = clamp(origin[0] + dx, 0, w); y1 = clamp(origin[1] + dy, 0, h);
      x2 = clamp(origin[2] + dx, 0, w); y2 = clamp(origin[3] + dy, 0, h);
    } else {
      if (corner.includes("w")) x1 = clamp(point.x, 0, w);
      if (corner.includes("e")) x2 = clamp(point.x, 0, w);
      if (corner.includes("n")) y1 = clamp(point.y, 0, h);
      if (corner.includes("s")) y2 = clamp(point.y, 0, h);
    }
    state.boxes[state.drag.index].bbox = [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)].map((v) => Math.round(v * 100) / 100);
    renderBoxes();
  }

  function onPointerUp() {
    if (!state.drag) return;
    const { moved, index } = state.drag;
    state.drag = null;
    if (!moved) {
      state.boxes[index].accepted = !state.boxes[index].accepted;
      renderBoxes();
    }
  }

  function startNewBox(event) {
    if (!state.addMode || event.button !== 0) return;
    event.preventDefault();
    const start = pointerToImage(event);
    state.drag = { index: state.boxes.length, corner: "se", start, origin: [start.x, start.y, start.x, start.y], moved: true, isNew: true };
    state.boxes.push({ bbox: [start.x, start.y, start.x, start.y], confidence: 0, accepted: true });
    state.active = state.boxes.length - 1;
    el.overlay.setPointerCapture(event.pointerId);
    renderBoxes();
  }

  function onOverlayPointerMove(event) {
    if (state.drag && state.drag.isNew) {
      const point = pointerToImage(event);
      const { start } = state.drag;
      state.boxes[state.drag.index].bbox = [
        Math.min(start.x, point.x), Math.min(start.y, point.y),
        Math.max(start.x, point.x), Math.max(start.y, point.y),
      ].map((v) => Math.round(v * 100) / 100);
      renderBoxes();
      return;
    }
    onPointerMove(event);
  }

  async function save() {
    const record = state.records[state.index];
    if (!record) return;
    const accepted = state.boxes
      .filter((box) => box.accepted)
      .map((box) => ({ bbox: box.bbox, class: "Stone", confidence: box.confidence || 0 }));
    try {
      const response = await fetch("/api/review/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: record.image, accepted }),
      });
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.message || "Gagal menyimpan");
      record.reviewed = true;
      record.accepted = accepted;
      setStatus(`Tersimpan: ${data.accepted} kotak.`, "ok");
      renderList();
    } catch (error) {
      setStatus(error.message, "err");
    }
  }

  function move(delta) {
    const visible = visibleIndices();
    const pos = visible.indexOf(state.index);
    const nextPos = clamp((pos === -1 ? 0 : pos) + delta, 0, visible.length - 1);
    select(visible[nextPos]);
  }

  el.image.addEventListener("load", renderBoxes);
  el.overlay.addEventListener("pointerdown", startNewBox);
  el.overlay.addEventListener("pointermove", onOverlayPointerMove);
  el.overlay.addEventListener("pointerup", onPointerUp);
  el.overlay.addEventListener("pointercancel", onPointerUp);

  el.add.addEventListener("click", () => {
    state.addMode = !state.addMode;
    el.add.classList.toggle("is-active", state.addMode);
    setStatus(state.addMode ? "Mode tambah: seret di area gambar untuk membuat kotak." : "");
  });

  el.auto.addEventListener("click", () => {
    state.autoTop1 = !state.autoTop1;
    localStorage.setItem("rv-auto-top1", state.autoTop1 ? "on" : "off");
    updateAutoLabel();
    if (state.index >= 0 && !state.records[state.index].reviewed) {
      state.boxes = loadBoxes(state.records[state.index]);
      renderBoxes();
    }
  });

  el.acceptAll.addEventListener("click", () => {
    state.boxes.forEach((box) => { box.accepted = true; });
    renderBoxes();
  });
  el.clear.addEventListener("click", () => {
    state.boxes = [];
    state.active = -1;
    renderBoxes();
  });
  el.prev.addEventListener("click", () => move(-1));
  el.next.addEventListener("click", () => move(1));
  el.save.addEventListener("click", save);

  const setFilter = (filter, button) => {
    state.filter = filter;
    [el.filterAll, el.filterTodo, el.filterDone].forEach((chip) => chip.classList.remove("is-active"));
    button.classList.add("is-active");
    renderList();
  };
  el.filterAll.addEventListener("click", () => setFilter("all", el.filterAll));
  el.filterTodo.addEventListener("click", () => setFilter("todo", el.filterTodo));
  el.filterDone.addEventListener("click", () => setFilter("done", el.filterDone));

  document.addEventListener("keydown", (event) => {
    if (event.target.tagName === "INPUT") return;
    if (event.key === "a" || event.key === "A") { state.boxes.forEach((box) => { box.accepted = true; }); renderBoxes(); }
    else if (event.key === "n" || event.key === "N") move(1);
    else if (event.key === "p" || event.key === "P") move(-1);
    else if (event.key === "Delete" && state.active >= 0) {
      state.boxes.splice(state.active, 1);
      state.active = -1;
      renderBoxes();
    } else if (event.key === "s" || event.key === "S") save();
  });

  async function init() {
    updateAutoLabel();
    try {
      const response = await fetch("/api/review/queue");
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.message || "Antrean tidak tersedia");
      state.records = data.queue.records;
      el.queuePath.textContent = data.queue.queue_path;
      renderList();
      if (state.records.length) select(0);
    } catch (error) {
      el.empty.hidden = false;
      el.empty.textContent = error.message;
      setStatus(error.message, "err");
    }
  }

  init();
})();
