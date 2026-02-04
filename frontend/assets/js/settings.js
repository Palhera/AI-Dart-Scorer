(() => {
  // =============================
  // Back navigation
  // =============================
  const backLink = document.querySelector("[data-back-link]");
  if (backLink) {
    backLink.addEventListener("click", (event) => {
      event.preventDefault();
      // Prefer browser history when possible; fall back to home for direct entry / deep links.
      if (window.history.length > 1) {
        window.history.back();
      } else {
        window.location.href = "/";
      }
    });
  }

  // =============================
  // General settings
  // =============================
  const toggle = document.getElementById("data-collection-toggle");
  const folderLine = document.getElementById("data-folder-line");
  const folderButton = document.getElementById("data-folder-button");
  const folderName = document.getElementById("data-folder-name");

  if (toggle) {
    const KEYS = {
      show: "aiDartScorer.settings.showDataCollection",
      folder: "aiDartScorer.settings.dataFolderName",
    };

    const readBool = (k, d = false) => (localStorage.getItem(k) ?? String(d)) === "true";
    const readText = (k, d = "") => localStorage.getItem(k) ?? d;

    const apply = (enabled) => {
      // Keep DOM state and ARIA state in sync for accessibility + predictable styling.
      toggle.checked = enabled;
      toggle.setAttribute("aria-checked", String(enabled));
      folderLine?.setAttribute("aria-disabled", String(!enabled));
      if (folderButton) folderButton.disabled = !enabled;
    };

    const setFolder = (name) => {
      const v = (name || "").trim();
      if (!folderName) return;
      folderName.textContent = v || "Not set";
      folderName.classList.toggle("is-set", Boolean(v));
    };

    apply(readBool(KEYS.show));
    setFolder(readText(KEYS.folder));

    toggle.addEventListener("change", () => {
      // This flag also drives visibility of the "Data Collection" game mode on the home screen.
      localStorage.setItem(KEYS.show, String(toggle.checked));
      apply(toggle.checked);
    });

    folderButton?.addEventListener("click", async () => {
      if (!readBool(KEYS.show)) return;

      // If the File System Access API is available, persist a directory handle (best UX).
      // Fallback: store a user-provided label (name only).
      if (window.showDirectoryPicker) {
        try {
          const h = await window.showDirectoryPicker();
          if (h?.name) {
            try {
              await window.aiDartScorerFS?.saveDirectoryHandle(h);
            } catch {}
            localStorage.setItem(KEYS.folder, h.name);
            setFolder(h.name);
          }
        } catch {}
      } else {
        const v = window.prompt("Enter a folder name");
        if (v !== null) {
          localStorage.setItem(KEYS.folder, v);
          setFolder(v);
        }
      }
    });
  }

  // =============================
  // Cameras (snapshots)
  // =============================
  const CAMS = ["cam1", "cam2", "cam3"];
  const snapshotUrls = new Map();
  const snapshotSeq = new Map();
  const tileReady = new Set();

  const setLiveUI = (camId, live) => {
    document
      .getElementById(camId)
      ?.closest(".cam-tile")
      ?.classList.toggle("is-live", live);

    const btn = document.querySelector(`.cam-card[data-cam="${camId}"] .calibrate-btn`);
    if (btn) btn.disabled = !live;
  };

  const snapshotUrl = (camId, format = "jpg") =>
    `/api/camera/snapshot/${camId}?format=${format}&t=${Date.now()}`;

  const fetchSnapshotBlob = async (camId, format = "jpg") => {
    const res = await fetch(snapshotUrl(camId, format), { cache: "no-store" });
    if (!res.ok) {
      let detail = `Snapshot failed (${res.status})`;
      try {
        const err = await res.json();
        if (err?.detail) detail = err.detail;
      } catch {}
      throw new Error(detail);
    }
    return res.blob();
  };

  const swapImage = (key, img, url, onReady) => {
    const prev = snapshotUrls.get(key);
    snapshotUrls.set(key, url);
    img.src = url;
    if (prev) URL.revokeObjectURL(prev);
    onReady?.(true);
  };

  const loadSnapshot = async (camId, img, onReady, options = {}) => {
    const { key = camId, keepLive = false } = options;
    if (!img) return false;

    const seq = (snapshotSeq.get(key) || 0) + 1;
    snapshotSeq.set(key, seq);

    if (!keepLive) onReady?.(false);

    let blob;
    try {
      blob = await fetchSnapshotBlob(camId, "jpg");
    } catch {
      if (!keepLive) onReady?.(false);
      return false;
    }

    const url = URL.createObjectURL(blob);
    return new Promise((resolve) => {
      const probe = new Image();
      probe.onload = () => {
        if (snapshotSeq.get(key) !== seq) {
          URL.revokeObjectURL(url);
          return resolve(false);
        }
        swapImage(key, img, url, onReady);
        resolve(true);
      };
      probe.onerror = () => {
        if (snapshotSeq.get(key) === seq && !keepLive) onReady?.(false);
        URL.revokeObjectURL(url);
        resolve(false);
      };
      probe.src = url;
    });
  };

  const loadTileSnapshot = (camId) =>
    loadSnapshot(
      camId,
      document.getElementById(camId),
      (ready) => {
        if (ready) tileReady.add(camId);
        setLiveUI(camId, ready || tileReady.has(camId));
      },
      { keepLive: tileReady.has(camId), key: `tile:${camId}` },
    );

  const waitBackendReady = async () => {
    // Backend "ready" becomes true even if some cameras are missing; per-camera UI still
    // relies on snapshot load success to reflect actual availability.
    while (true) {
      try {
        const r = await fetch("/api/status", { cache: "no-store" });
        if ((await r.json())?.ready) break;
      } catch {}
      await new Promise((r) => setTimeout(r, 500));
    }
    await Promise.all(CAMS.map(loadTileSnapshot));
  };

  waitBackendReady();

  // =============================
  // Calibration modal
  // =============================
  const modal = document.getElementById("calibration-modal");
  const modalImg = document.getElementById("modal-image");
  const modalTile = document.getElementById("modal-preview-tile");
  const modalLabel = document.getElementById("modal-cam-label");
  const modalTitle = document.getElementById("calibration-title");
  const referenceOverlay = document.getElementById("modal-reference-overlay");

  if (!modal || !modalImg || !modalTile) return;

  const setModalLive = (live) => modalTile.classList.toggle("is-live", live);
  let modalReady = false;

  const loadModalSnapshot = (camId) =>
    loadSnapshot(
      camId,
      modalImg,
      (ready) => {
        if (ready) modalReady = true;
        setModalLive(ready || modalReady);
      },
      { keepLive: modalReady, key: `modal:${camId}` },
    );

  const refreshSnapshots = async (camId, options = {}) => {
    const { includeModal = true } = options;
    const tasks = [loadTileSnapshot(camId)];
    if (includeModal && modal.classList.contains("is-open") && currentCam === camId) {
      tasks.push(loadModalSnapshot(camId));
    }
    await Promise.all(tasks);
  };

  const REF = {
    // Mirror of backend reference constants; keep in sync with backend/vision/reference.py.
    // Used only for drawing UI overlays and defining the default manual-warp quad.
    board: 451.0,
    outer: 170.0,
    inner: 15.9,
    rings: [170.0, 162.0, 107.0, 99.0, 15.9, 6.35],
    rot: -(9 * Math.PI) / 180,
    angles: Array.from({ length: 10 }, (_, i) => (i * Math.PI) / 10),
    corners: [225, 315, 45, 135].map((d) => (d * Math.PI) / 180),
  };

  let currentCam = null;
  let overlayDrawPending = false;
  let manualQuad = null;
  let manualApplyPending = false;
  const drag = { active: null, pointerId: null, last: null, changed: false };

  const refBuffer = document.createElement("canvas");
  let refBufferSize = { w: 0, h: 0 };

  const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
  const geom = (w, h) => {
    const size = Math.min(w, h) - 1;
    const scale = size / REF.board;
    return { size, scale, outer: REF.outer * scale, cx: (w - 1) * 0.5, cy: (h - 1) * 0.5 };
  };
  const refQuad = (w, h) => {
    // Source quad is defined on the reference circle at fixed angles (matches backend manual-warp source).
    const g = geom(w, h);
    if (g.size <= 0) return null;
    return REF.corners.map((t) => ({ x: g.cx + g.outer * Math.cos(t), y: g.cy + g.outer * Math.sin(t) }));
  };
  const defaultQuad = (w, h) => {
    // Store quad normalized to [0,1] so it survives canvas resizes without recomputing.
    const q = refQuad(w, h);
    return q ? q.map((p) => ({ x: p.x / w, y: p.y / h })) : null;
  };
  const resetQuad = (w, h) => (manualQuad = defaultQuad(w, h));
  const ensureQuad = (w, h) => {
    if (!manualQuad) manualQuad = defaultQuad(w, h);
  };
  const quadPx = (w, h) => manualQuad?.map((p) => ({ x: p.x * w, y: p.y * h })) ?? [];
  const mid = (a, b) => ({ x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 });
  const projectToCircle = (p, g) => {
    // Edge handles are constrained to the reference circle for more intuitive "drag the rim" behavior.
    const dx = p.x - g.cx;
    const dy = p.y - g.cy;
    const len = Math.hypot(dx, dy) || 1;
    return { x: g.cx + (dx / len) * g.outer, y: g.cy + (dy / len) * g.outer };
  };
  const edgeHandles = (q, g) =>
    [mid(q[0], q[1]), mid(q[1], q[2]), mid(q[2], q[3]), mid(q[3], q[0])].map((p) =>
      projectToCircle(p, g),
    );
  const clampQuad = () => {
    // Prevent handle inversion (keeps the quad convex-ish and avoids degenerate transforms).
    if (!manualQuad) return;
    const g = 0.03;
    manualQuad[0].x = clamp(manualQuad[0].x, 0, manualQuad[1].x - g);
    manualQuad[1].x = clamp(manualQuad[1].x, manualQuad[0].x + g, 1);
    manualQuad[3].x = clamp(manualQuad[3].x, 0, manualQuad[2].x - g);
    manualQuad[2].x = clamp(manualQuad[2].x, manualQuad[3].x + g, 1);
    manualQuad[0].y = clamp(manualQuad[0].y, 0, manualQuad[3].y - g);
    manualQuad[3].y = clamp(manualQuad[3].y, manualQuad[0].y + g, 1);
    manualQuad[1].y = clamp(manualQuad[1].y, 0, manualQuad[2].y - g);
    manualQuad[2].y = clamp(manualQuad[2].y, manualQuad[1].y + g, 1);
  };
  const handleSizes = (size) => ({ edge: Math.max(6, size / 110), corner: Math.max(4, size / 140) });

  const hitHandle = (x, y, w, h) => {
    if (!manualQuad) return null;
    const g = geom(w, h);
    const { edge, corner } = handleSizes(g.size);
    const q = quadPx(w, h);
    const e = edgeHandles(q, g);
    for (let i = 0; i < q.length; i += 1) {
      if (Math.hypot(x - q[i].x, y - q[i].y) <= corner) return { type: "corner", index: i };
    }
    for (let i = 0; i < e.length; i += 1) {
      if (Math.hypot(x - e[i].x, y - e[i].y) <= edge) return { type: "edge", index: i };
    }
    return null;
  };
  const setCursor = (hit) => {
    if (!referenceOverlay) return;
    if (drag.active || manualApplyPending) return (referenceOverlay.style.cursor = "grabbing");
    if (!hit) return (referenceOverlay.style.cursor = "default");
    if (hit.type === "edge")
      return (referenceOverlay.style.cursor = hit.index % 2 === 0 ? "ns-resize" : "ew-resize");
    referenceOverlay.style.cursor = hit.index % 2 === 0 ? "nwse-resize" : "nesw-resize";
  };
  const applyDrag = (dx, dy) => {
    if (!manualQuad || !drag.active) return;
    if (drag.active.type === "edge") {
      const map = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
      ][drag.active.index];
      map.forEach((i) => {
        manualQuad[i].x += dx;
        manualQuad[i].y += dy;
      });
    } else {
      const i = drag.active.index;
      manualQuad[i].x += dx;
      manualQuad[i].y += dy;
    }
    clampQuad();
  };

  const drawBase = (ctx, w, h) => {
    // Draw the canonical rings and radial lines in the overlay canvas coordinate space.
    const g = geom(w, h);
    if (g.size <= 0) return;
    ctx.strokeStyle = "rgba(0, 229, 255, 0.92)";
    ctx.lineWidth = Math.max(1.2, g.size / 600);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const r of REF.rings) {
      const rp = r * g.scale;
      if (rp > 0) {
        ctx.beginPath();
        ctx.arc(g.cx, g.cy, rp, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    const inner = REF.inner * g.scale;
    const outer = REF.outer * g.scale;
    for (const a of REF.angles) {
      const t = a + REF.rot;
      const dx = Math.cos(t);
      const dy = Math.sin(t);
      ctx.beginPath();
      ctx.moveTo(g.cx + dx * inner, g.cy + dy * inner);
      ctx.lineTo(g.cx + dx * outer, g.cy + dy * outer);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(g.cx - dx * inner, g.cy - dy * inner);
      ctx.lineTo(g.cx - dx * outer, g.cy - dy * outer);
      ctx.stroke();
    }
  };

  const ensureRefBuffer = (w, h) => {
    // Cache the base reference drawing to avoid re-rendering it for every animation frame.
    if (refBufferSize.w === w && refBufferSize.h === h) return;
    refBufferSize = { w, h };
    refBuffer.width = w;
    refBuffer.height = h;
    const rctx = refBuffer.getContext("2d");
    if (!rctx) return;
    rctx.setTransform(1, 0, 0, 1, 0, 0);
    rctx.clearRect(0, 0, w, h);
    drawBase(rctx, w, h);
  };

  const drawTri = (ctx, img, s0, s1, s2, d0, d1, d2) => {
    // Piecewise-affine drawing: approximate a perspective warp by subdividing into triangles.
    // This is for preview only; the real warp is applied server-side via homography matrices.
    const denom = s0.x * (s1.y - s2.y) + s1.x * (s2.y - s0.y) + s2.x * (s0.y - s1.y);
    if (Math.abs(denom) < 1e-6) return;
    const a = (d0.x * (s1.y - s2.y) + d1.x * (s2.y - s0.y) + d2.x * (s0.y - s1.y)) / denom;
    const b = (d0.y * (s1.y - s2.y) + d1.y * (s2.y - s0.y) + d2.y * (s0.y - s1.y)) / denom;
    const c = (d0.x * (s2.x - s1.x) + d1.x * (s0.x - s2.x) + d2.x * (s1.x - s0.x)) / denom;
    const d = (d0.y * (s2.x - s1.x) + d1.y * (s0.x - s2.x) + d2.y * (s1.x - s0.x)) / denom;
    const e =
      (d0.x * (s1.x * s2.y - s2.x * s1.y) +
        d1.x * (s2.x * s0.y - s0.x * s2.y) +
        d2.x * (s0.x * s1.y - s1.x * s0.y)) /
      denom;
    const f =
      (d0.y * (s1.x * s2.y - s2.x * s1.y) +
        d1.y * (s2.x * s0.y - s0.x * s2.y) +
        d2.y * (s0.x * s1.y - s1.x * s0.y)) /
      denom;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(d0.x, d0.y);
    ctx.lineTo(d1.x, d1.y);
    ctx.lineTo(d2.x, d2.y);
    ctx.closePath();
    ctx.clip();
    ctx.setTransform(a, b, c, d, e, f);
    ctx.drawImage(img, 0, 0);
    ctx.restore();
  };

  const solve = (A, b) => {
    // Minimal Gaussian elimination for 8x8 system (homography with h33 fixed to 1).
    const n = b.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let i = 0; i < n; i += 1) {
      let p = i;
      let m = Math.abs(M[i][i]);
      for (let r = i + 1; r < n; r += 1) {
        const v = Math.abs(M[r][i]);
        if (v > m) {
          m = v;
          p = r;
        }
      }
      if (m < 1e-9) return null;
      if (p !== i) [M[i], M[p]] = [M[p], M[i]];
      const pv = M[i][i];
      for (let c = i; c <= n; c += 1) M[i][c] /= pv;
      for (let r = 0; r < n; r += 1) {
        if (r === i) continue;
        const f = M[r][i];
        if (!f) continue;
        for (let c = i; c <= n; c += 1) M[r][c] -= f * M[i][c];
      }
    }
    return M.map((row) => row[n]);
  };

  const homography = (src, dst) => {
    // Compute H such that dst ~ H * src, with H[2][2] fixed to 1.
    if (src.length !== 4 || dst.length !== 4) return null;
    const A = [];
    const b = [];
    for (let i = 0; i < 4; i += 1) {
      const s = src[i];
      const d = dst[i];
      A.push([s.x, s.y, 1, 0, 0, 0, -d.x * s.x, -d.x * s.y]);
      b.push(d.x);
      A.push([0, 0, 0, s.x, s.y, 1, -d.y * s.x, -d.y * s.y]);
      b.push(d.y);
    }
    const h = solve(A, b);
    return h
      ? [
          [h[0], h[1], h[2]],
          [h[3], h[4], h[5]],
          [h[6], h[7], 1],
        ]
      : null;
  };

  const warp = (H, x, y) => {
    const d = H[2][0] * x + H[2][1] * y + H[2][2];
    if (Math.abs(d) < 1e-6) return null;
    return { x: (H[0][0] * x + H[0][1] * y + H[0][2]) / d, y: (H[1][0] * x + H[1][1] * y + H[1][2]) / d };
  };

  const drawWarped = (ctx, w, h, srcQ, dstQ) => {
    // Preview the effect of the current manual quad by warping the reference overlay
    // into the user's quad. This provides immediate feedback while dragging.
    ensureRefBuffer(w, h);
    const H = homography(srcQ, dstQ);
    if (!H) return drawBase(ctx, w, h);
    const cols = 14;
    const rows = 14;
    ctx.imageSmoothingEnabled = true;
    for (let iy = 0; iy < rows; iy += 1) {
      const y0 = (h * iy) / rows;
      const y1 = (h * (iy + 1)) / rows;
      for (let ix = 0; ix < cols; ix += 1) {
        const x0 = (w * ix) / cols;
        const x1 = (w * (ix + 1)) / cols;
        const s0 = { x: x0, y: y0 };
        const s1 = { x: x1, y: y0 };
        const s2 = { x: x1, y: y1 };
        const s3 = { x: x0, y: y1 };
        const d0 = warp(H, s0.x, s0.y);
        const d1 = warp(H, s1.x, s1.y);
        const d2 = warp(H, s2.x, s2.y);
        const d3 = warp(H, s3.x, s3.y);
        if (!d0 || !d1 || !d2 || !d3) continue;
        drawTri(ctx, refBuffer, s0, s1, s2, d0, d1, d2);
        drawTri(ctx, refBuffer, s0, s2, s3, d0, d2, d3);
      }
    }
  };

  const scheduleDraw = () => {
    // Coalesce multiple updates (drag/move/resize) into a single animation frame.
    if (!referenceOverlay || overlayDrawPending) return;
    overlayDrawPending = true;
    requestAnimationFrame(() => {
      overlayDrawPending = false;
      drawOverlay();
    });
  };

  const drawOverlay = () => {
    if (!referenceOverlay || !modal.classList.contains("is-open")) return;
    const ctx = referenceOverlay.getContext("2d");
    if (!ctx) return;
    const rect = modalTile.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    const dpr = window.devicePixelRatio || 1;
    const wd = Math.max(1, Math.round(w * dpr));
    const hd = Math.max(1, Math.round(h * dpr));
    referenceOverlay.width = wd;
    referenceOverlay.height = hd;
    referenceOverlay.style.width = `${w}px`;
    referenceOverlay.style.height = `${h}px`;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, wd, hd);

    ensureQuad(w, h);
    if (!manualQuad) return;

    const g = geom(wd, hd);
    if (g.size <= 0) return;

    const q = quadPx(wd, hd);
    const e = edgeHandles(q, g);
    const { edge, corner } = handleSizes(g.size);

    // While dragging, show the warped overlay preview; otherwise show the static reference overlay.
    if (drag.active || manualApplyPending) {
      const srcQ = refQuad(wd, hd);
      if (srcQ) drawWarped(ctx, wd, hd, srcQ, q);
    } else {
      drawBase(ctx, wd, hd);
    }

    const drawHandle = (p, r, fill) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = Math.max(1, r * 0.5);
      ctx.strokeStyle = "rgba(0, 0, 0, 0.6)";
      ctx.stroke();
    };
    e.forEach((p) => drawHandle(p, edge, "rgba(255, 203, 64, 0.95)"));
    q.forEach((p) => drawHandle(p, corner, "rgba(255, 255, 255, 0.92)"));
  };

  const callBackend = (camId, action) =>
    fetch("/api/camera/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cam_id: camId, action }),
    });

  const applyManualWarp = async () => {
    // Persist the manual correction by posting normalized quad points to the backend.
    // Backend composes this correction with the existing homography and saves it on disk.
    if (!currentCam || !manualQuad || manualApplyPending) return;
    manualApplyPending = true;
    try {
      const points = manualQuad.map((p) => ({ x: clamp(p.x, 0, 1), y: clamp(p.y, 0, 1) }));
      const res = await fetch("/api/camera/manual-warp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cam_id: currentCam, points }),
      });
      if (!res.ok) {
        let detail = `Request failed (${res.status})`;
        try {
          const err = await res.json();
          if (err?.detail) detail = err.detail;
        } catch {}
        throw new Error(detail);
      }
      const rect = modalTile.getBoundingClientRect();
      resetQuad(rect.width, rect.height);
      scheduleDraw();
      await refreshSnapshots(currentCam);
    } catch (err) {
      console.warn(err);
    } finally {
      manualApplyPending = false;
    }
  };

  const pointerInfo = (event) => {
    if (!referenceOverlay) return null;
    const rect = referenceOverlay.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    return { x, y, nx: x / rect.width, ny: y / rect.height, width: rect.width, height: rect.height };
  };

  const openModal = (camId) => {
    currentCam = camId;
    modalLabel.textContent = camId.toUpperCase();
    modalTitle.textContent = `Calibration - ${camId.toUpperCase()}`;
    modalReady = false;
    setModalLive(false);
    loadModalSnapshot(camId);
    const rect = modalTile.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) resetQuad(rect.width, rect.height);
    else manualQuad = null;
    modal.classList.add("is-open");
    modal.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
    scheduleDraw();
  };

  const closeModal = () => {
    const closingCam = currentCam;
    currentCam = null;
    modal.classList.remove("is-open");
    modal.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
    modalImg.removeAttribute("src");
    modalTile.classList.remove("is-live");
    modalReady = false;
    if (closingCam) {
      const key = `modal:${closingCam}`;
      const prev = snapshotUrls.get(key);
      if (prev) {
        URL.revokeObjectURL(prev);
        snapshotUrls.delete(key);
      }
    }
    manualQuad = null;
    drag.active = null;
    drag.pointerId = null;
    drag.last = null;
    drag.changed = false;
  };

  document.addEventListener("click", async (e) => {
    // Single delegated handler keeps wiring simple even if DOM is re-rendered.
    const openBtn = e.target.closest("[data-open-calibration]");
    if (openBtn) return openModal(openBtn.dataset.openCalibration);
    if (e.target.closest("[data-close-modal]")) return closeModal();
    const actionBtn = e.target.closest("[data-modal-action]");
    if (actionBtn && currentCam) {
      const action = actionBtn.dataset.modalAction;
      actionBtn.disabled = true;
      try {
        await callBackend(currentCam, action);
        if (action === "calibrate" || action === "reset") {
          const rect = modalTile.getBoundingClientRect();
          if (rect.width > 0 && rect.height > 0) resetQuad(rect.width, rect.height);
          else manualQuad = null;
          scheduleDraw();
        }
        // Refresh snapshots after any calibration change.
        await refreshSnapshots(currentCam);
      } finally {
        actionBtn.disabled = false;
      }
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.classList.contains("is-open")) closeModal();
  });

  if (referenceOverlay) {
    referenceOverlay.addEventListener("pointerdown", (event) => {
      if (!modal.classList.contains("is-open")) return;
      const info = pointerInfo(event);
      if (!info) return;
      ensureQuad(info.width, info.height);
      const hit = hitHandle(info.x, info.y, info.width, info.height);
      setCursor(hit);
      if (!hit) return;
      drag.active = hit;
      drag.pointerId = event.pointerId;
      drag.last = { x: info.nx, y: info.ny };
      drag.changed = false;
      referenceOverlay.setPointerCapture(event.pointerId);
      setCursor(hit);
      event.preventDefault();
    });

    referenceOverlay.addEventListener("pointermove", (event) => {
      if (!modal.classList.contains("is-open")) return;
      const info = pointerInfo(event);
      if (!info) return;
      if (!drag.active || drag.pointerId !== event.pointerId) {
        setCursor(hitHandle(info.x, info.y, info.width, info.height));
        return;
      }
      const dx = info.nx - drag.last.x;
      const dy = info.ny - drag.last.y;
      if (dx || dy) {
        applyDrag(dx, dy);
        drag.last = { x: info.nx, y: info.ny };
        drag.changed = true;
        scheduleDraw();
      }
      event.preventDefault();
    });

    const endDrag = (event) => {
      if (!drag.active || drag.pointerId !== event.pointerId) return;
      if (referenceOverlay.hasPointerCapture(event.pointerId)) {
        referenceOverlay.releasePointerCapture(event.pointerId);
      }
      const changed = drag.changed;
      drag.active = null;
      drag.pointerId = null;
      drag.last = null;
      drag.changed = false;
      setCursor(null);
      // Apply on pointer release to avoid spamming the backend while dragging.
      if (changed) void applyManualWarp();
      event.preventDefault();
    };

    referenceOverlay.addEventListener("pointerup", endDrag);
    referenceOverlay.addEventListener("pointercancel", endDrag);
    referenceOverlay.addEventListener("pointerleave", () => {
      if (!drag.active) setCursor(null);
    });
  }

  if (referenceOverlay && "ResizeObserver" in window) {
    const observer = new ResizeObserver(() => {
      if (modal.classList.contains("is-open")) scheduleDraw();
    });
    observer.observe(modalTile);
  } else {
    window.addEventListener("resize", () => {
      if (modal.classList.contains("is-open")) scheduleDraw();
    });
  }

  // =============================
  // TEMP DEBUG: Homography upload (REMOVE BEFORE RELEASE)
  // =============================
  const debugInput = document.getElementById("debug-image-input");
  const debugRun = document.getElementById("debug-run-btn");
  const debugImg = document.getElementById("debug-result-image");
  const debugMatrix = document.getElementById("debug-matrix");
  const debugStatus = document.getElementById("debug-status");

  if (debugInput && debugRun && debugImg && debugMatrix && debugStatus) {
    const setDebugStatus = (msg, isError = false) => {
      debugStatus.textContent = msg;
      debugStatus.classList.toggle("is-error", isError);
    };

    const renderMatrix = (matrix) => {
      if (!matrix) {
        debugMatrix.textContent = "Matrix: null";
        return;
      }
      debugMatrix.textContent = JSON.stringify(matrix, null, 2);
    };

    const runDebug = async () => {
      const file = debugInput.files?.[0];
      if (!file) {
        setDebugStatus("Choose an image first.", true);
        return;
      }

      debugRun.disabled = true;
      setDebugStatus("Processing...");
      try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch("/keypoints", { method: "POST", body: formData });
        if (!res.ok) {
          let detail = `Request failed (${res.status})`;
          try {
            const err = await res.json();
            if (err?.detail) detail = err.detail;
          } catch {}
          throw new Error(detail);
        }

        const data = await res.json();
        if (data?.image) {
          debugImg.src = `data:image/png;base64,${data.image}`;
        }
        renderMatrix(data?.total_warp_matrix);
        setDebugStatus("Done.");
      } catch (err) {
        setDebugStatus(err?.message || "Failed to compute homography.", true);
      } finally {
        debugRun.disabled = false;
      }
    };

    debugRun.addEventListener("click", runDebug);
    debugInput.addEventListener("change", () => {
      if (debugInput.files?.length) {
        setDebugStatus("");
      }
    });
  }
})();
