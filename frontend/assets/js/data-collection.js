(() => {
  const CAMS = ["cam1", "cam2", "cam3"];
  const FOLDER_KEY = "aiDartScorer.settings.dataFolderName";

  const captureBtn = document.getElementById("dc-capture-btn");
  const statusEl = document.getElementById("dc-status");
  const folderEl = document.getElementById("dc-folder-name");
  const nextEl = document.getElementById("dc-next-id");

  if (!captureBtn) return;

  let dirHandle = null;
  let usedIds = new Set();
  let nextId = 1;
  let saving = false;
  let camerasReady = false;
  let folderReady = false;
  let folderMissing = true;

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  const setStatus = (message, state = "info") => {
    // statusEl.dataset.state is used for styling (ok/info/busy/error).
    if (!statusEl) return;
    statusEl.textContent = message;
    statusEl.dataset.state = state;
  };

  const setFolderName = (name) => {
    if (!folderEl) return;
    const value = (name || "").trim();
    folderEl.textContent = value || "Not set";
    folderEl.classList.toggle("is-set", Boolean(value));
  };

  const formatId = (id) => {
    // Zero-pad to keep filenames sortable by lexicographic order.
    const value = String(id);
    return value.length >= 5 ? value : value.padStart(5, "0");
  };

  const setNextId = (id) => {
    if (!nextEl) return;
    nextEl.textContent = formatId(id);
  };

  const updateCaptureState = () => {
    // Capture is only allowed when:
    // - not already saving
    // - folder exists and is accessible
    // - all camera streams are live (so canvas capture has pixels)
    captureBtn.disabled = saving || folderMissing || !camerasReady;
  };

  const setLiveUI = (camId, live) => {
    const tile = document.querySelector(`.dc-cam-card[data-cam="${camId}"] .dc-cam-tile`);
    if (tile) tile.classList.toggle("is-live", live);
  };

  const setStream = (img, camId) => {
    // Cache-bust to force reconnection and avoid stale frames.
    img.src = `/api/stream/${camId}?t=${Date.now()}`;
  };

  const attachStream = (camId, liveState, onLive) => {
    const img = document.getElementById(`dc-${camId}`);
    if (!img) return;

    setLiveUI(camId, false);
    setStream(img, camId);

    img.onload = () => {
      setLiveUI(camId, true);
      // Count the first successful load per camera to know when all streams are ready.
      if (!liveState[camId]) {
        liveState[camId] = true;
        onLive();
      }
    };

    img.onerror = () => {
      setLiveUI(camId, false);
      setTimeout(() => setStream(img, camId), 1000);
    };
  };

  const waitBackendReady = async () => {
    // Wait for backend startup (camera threads initialized) before attaching MJPEG streams.
    while (true) {
      try {
        const r = await fetch("/api/status", { cache: "no-store" });
        if ((await r.json())?.ready) return;
      } catch {}
      await new Promise((r) => setTimeout(r, 500));
    }
  };

  const ensureFolderHandle = async (requestPermission = true) => {
    // Uses a small abstraction layer (window.aiDartScorerFS) so the same UI can run:
    // - in browsers supporting File System Access API
    // - or in an app shell that provides a compatible bridge.
    const fsApi = window.aiDartScorerFS;
    if (!fsApi?.loadDirectoryHandle) {
      folderMissing = true;
      folderReady = false;
      updateCaptureState();
      setStatus("File system access is not supported in this browser.", "error");
      return null;
    }

    if (!dirHandle) {
      try {
        dirHandle = await fsApi.loadDirectoryHandle();
      } catch {}
    }

    if (!dirHandle) {
      folderMissing = true;
      folderReady = false;
      updateCaptureState();
      setStatus("Data folder not set. Open Settings to choose a folder.", "error");
      return null;
    }

    try {
      folderMissing = false;
      const current = await dirHandle.queryPermission({ mode: "readwrite" });
      if (current !== "granted") {
        // On some browsers, write permission must be explicitly granted per handle.
        if (!requestPermission) {
          folderReady = false;
          updateCaptureState();
          setStatus("Folder access needed. Press Capture to grant permission.", "info");
          return null;
        }
        const ok = await fsApi.ensurePermission(dirHandle, "readwrite");
        if (!ok) {
          folderReady = false;
          updateCaptureState();
          setStatus("Folder permission denied. Re-select it in Settings.", "error");
          return null;
        }
      }
    } catch {
      folderReady = false;
      updateCaptureState();
      setStatus("Unable to access the selected folder.", "error");
      return null;
    }

    return dirHandle;
  };

  const findNextId = (set, start) => {
    // IDs are monotonically increased, skipping any already present on disk.
    let id = Math.max(1, start);
    while (set.has(id)) id += 1;
    return id;
  };

  const scanExistingIds = async () => {
    // Build an index of already-captured sample IDs by scanning the folder contents.
    // Filename convention: "00001_cam1.png" etc. (cam suffix is used in the regex).
    const handle = await ensureFolderHandle(false);
    if (!handle) return;

    const ids = new Set();
    const re = /^(\d+)_cam([1-3])\.png$/i;

    try {
      for await (const entry of handle.values()) {
        if (!entry || entry.kind !== "file") continue;
        const match = entry.name.match(re);
        if (!match) continue;
        const id = Number.parseInt(match[1], 10);
        if (Number.isFinite(id) && id > 0) ids.add(id);
      }
    } catch {
      folderReady = false;
      updateCaptureState();
      setStatus("Unable to read the data folder.", "error");
      return;
    }

    usedIds = ids;
    nextId = findNextId(usedIds, 1);
    setNextId(nextId);

    folderReady = true;
    updateCaptureState();
    if (camerasReady) {
      setStatus("Ready to capture.", "ok");
    } else {
      setStatus("Waiting for camera streams...", "info");
    }
  };

  const grabPngFromImage = async (img) => {
    // Captures the current <img> contents by drawing to canvas, then encoding PNG.
    // This is "best effort": it captures what the browser currently has decoded.
    if (!ctx) throw new Error("Canvas is not available.");
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    if (!w || !h) throw new Error("Camera stream not ready.");
    canvas.width = w;
    canvas.height = h;
    ctx.drawImage(img, 0, 0, w, h);
    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (!blob) {
          reject(new Error("Failed to encode PNG."));
          return;
        }
        resolve(blob);
      }, "image/png");
    });
  };

  const writeBlob = async (handle, fileName, blob) => {
    // Atomic-ish write: createWritable() will replace the file contents.
    const fileHandle = await handle.getFileHandle(fileName, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(blob);
    await writable.close();
  };

  const captureImages = async () => {
    if (saving) return;
    saving = true;
    updateCaptureState();

    try {
      const handle = await ensureFolderHandle();
      if (!handle) return;
      if (!folderReady) {
        await scanExistingIds();
        if (!folderReady) return;
      }
      if (!camerasReady) {
        setStatus("Camera streams not ready.", "error");
        return;
      }

      setStatus("Saving images...", "busy");

      const id = nextId;
      const filePrefix = formatId(id);

      // Save 3 synchronized-ish samples (one per camera). Each is a browser-side capture
      // of the current MJPEG frame, not a backend timestamped snapshot.
      for (const camId of CAMS) {
        const img = document.getElementById(`dc-${camId}`);
        if (!img) throw new Error(`Missing stream for ${camId}.`);
        const blob = await grabPngFromImage(img);
        await writeBlob(handle, `${filePrefix}_${camId}.png`, blob);
      }

      usedIds.add(id);
      nextId = findNextId(usedIds, id + 1);
      setNextId(nextId);
      setStatus(`Saved ${filePrefix} for 3 cameras.`, "ok");
    } catch (err) {
      setStatus(err?.message || "Failed to save images.", "error");
    } finally {
      saving = false;
      updateCaptureState();
    }
  };

  const isInteractiveTarget = (target) => {
    // Prevent spacebar-to-capture from interfering with form controls and links.
    if (!target || !target.tagName) return false;
    if (target.isContentEditable) return true;
    const tag = target.tagName.toUpperCase();
    return (
      tag === "INPUT" ||
      tag === "TEXTAREA" ||
      tag === "SELECT" ||
      tag === "BUTTON" ||
      tag === "A"
    );
  };

  captureBtn.addEventListener("click", captureImages);

  document.addEventListener("keydown", (event) => {
    // Convenience shortcut: press Space anywhere (except interactive controls) to capture.
    if (event.code !== "Space" || event.repeat) return;
    if (isInteractiveTarget(event.target)) return;
    event.preventDefault();
    captureImages();
  });

  const folderName = localStorage.getItem(FOLDER_KEY) || "";
  setFolderName(folderName);

  waitBackendReady().then(() => {
    const liveState = {};
    let liveCount = 0;
    const onLive = () => {
      liveCount += 1;
      if (liveCount >= CAMS.length) {
        camerasReady = true;
        updateCaptureState();
        if (folderReady) setStatus("Ready to capture.", "ok");
      }
    };
    CAMS.forEach((camId) => attachStream(camId, liveState, onLive));
  });

  scanExistingIds();
})();
