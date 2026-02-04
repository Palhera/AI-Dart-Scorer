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

  const previewUrls = new Map();
  const previewSeq = new Map();
  const previewReady = new Set();

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
    // - camera snapshots have loaded at least once
    captureBtn.disabled = saving || folderMissing || !camerasReady;
  };

  const setLiveUI = (camId, live) => {
    const tile = document.querySelector(`.dc-cam-card[data-cam="${camId}"] .dc-cam-tile`);
    if (tile) tile.classList.toggle("is-live", live);
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

  const setPreviewBlob = (camId, blob) =>
    new Promise((resolve) => {
      const img = document.getElementById(`dc-${camId}`);
      if (!img) return resolve(false);

      const seq = (previewSeq.get(camId) || 0) + 1;
      previewSeq.set(camId, seq);

      const url = URL.createObjectURL(blob);
      const probe = new Image();
      probe.onload = () => {
        if (previewSeq.get(camId) !== seq) {
          URL.revokeObjectURL(url);
          return resolve(false);
        }
        const prev = previewUrls.get(camId);
        previewUrls.set(camId, url);
        img.src = url;
        if (prev) URL.revokeObjectURL(prev);
        previewReady.add(camId);
        setLiveUI(camId, true);
        resolve(true);
      };
      probe.onerror = () => {
        if (previewSeq.get(camId) === seq && !previewReady.has(camId)) {
          setLiveUI(camId, false);
        }
        URL.revokeObjectURL(url);
        resolve(false);
      };
      probe.src = url;
    });

  const loadPreviewSnapshot = async (camId) => {
    if (!previewReady.has(camId)) setLiveUI(camId, false);
    const blob = await fetchSnapshotBlob(camId, "jpg");
    return setPreviewBlob(camId, blob);
  };

  const loadAllPreviews = async () => {
    const results = await Promise.allSettled(CAMS.map(loadPreviewSnapshot));
    camerasReady = results.every((r) => r.status === "fulfilled");
    updateCaptureState();
    if (folderReady) {
      setStatus(
        camerasReady ? "Ready to capture." : "Camera snapshots not ready.",
        camerasReady ? "ok" : "error",
      );
    }
  };

  const waitBackendReady = async () => {
    // Wait for backend startup before requesting snapshots.
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
      setStatus("Waiting for camera snapshots...", "info");
    }
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
        setStatus("Camera snapshots not ready.", "error");
        return;
      }

      setStatus("Saving images...", "busy");

      const id = nextId;
      const filePrefix = formatId(id);

      // Save 3 synchronized-ish samples (one per camera) via backend snapshots.
      const snapshots = await Promise.all(
        CAMS.map(async (camId) => ({
          camId,
          blob: await fetchSnapshotBlob(camId, "png"),
        })),
      );

      for (const { camId, blob } of snapshots) {
        await writeBlob(handle, `${filePrefix}_${camId}.png`, blob);
        await setPreviewBlob(camId, blob);
      }

      camerasReady = true;
      updateCaptureState();

      usedIds.add(id);
      nextId = findNextId(usedIds, id + 1);
      setNextId(nextId);
      setStatus(`Saved ${filePrefix} for 3 cameras.`, "ok");
    } catch (err) {
      camerasReady = false;
      updateCaptureState();
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

  waitBackendReady().then(async () => {
    if (!folderMissing) setStatus("Loading camera snapshots...", "info");
    await loadAllPreviews();
  });

  scanExistingIds();
})();
