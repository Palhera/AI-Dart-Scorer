(() => {
  // =============================
  // Back navigation
  // =============================
  const backLink = document.querySelector("[data-back-link]");
  if (backLink) {
    backLink.addEventListener("click", (event) => {
      event.preventDefault();
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
      localStorage.setItem(KEYS.show, String(toggle.checked));
      apply(toggle.checked);
    });

    folderButton?.addEventListener("click", async () => {
      if (!readBool(KEYS.show)) return;

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
  // Cameras (MJPEG backend)
  // =============================
  const CAMS = ["cam1", "cam2", "cam3"];

  const setLiveUI = (camId, live) => {
    document
      .getElementById(camId)
      ?.closest(".cam-tile")
      ?.classList.toggle("is-live", live);

    const btn = document.querySelector(`.cam-card[data-cam="${camId}"] .calibrate-btn`);
    if (btn) btn.disabled = !live;
  };

  const setStream = (img, camId) => {
    img.src = `/api/stream/${camId}?t=${Date.now()}`;
  };

  const attachStream = (camId) => {
    const img = document.getElementById(camId);
    if (!img) return;

    setLiveUI(camId, false);
    setStream(img, camId);

    img.onload = () => setLiveUI(camId, true);
    img.onerror = () => {
      setLiveUI(camId, false);
      setTimeout(() => setStream(img, camId), 1000);
    };
  };

  const waitBackendReady = async () => {
    while (true) {
      try {
        const r = await fetch("/api/status", { cache: "no-store" });
        if ((await r.json())?.ready) break;
      } catch {}
      await new Promise((r) => setTimeout(r, 500));
    }
    CAMS.forEach(attachStream);
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

  let currentCam = null;
  let overlayDrawPending = false;

  const callBackend = (camId, action) =>
    fetch("/api/camera/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cam_id: camId, action }),
    });

  const scheduleOverlayDraw = () => {
    if (!referenceOverlay || overlayDrawPending) return;
    overlayDrawPending = true;
    requestAnimationFrame(() => {
      overlayDrawPending = false;
      drawReferenceOverlay();
    });
  };

  const drawReferenceOverlay = () => {
    if (!referenceOverlay || !modal.classList.contains("is-open")) return;
    const ctx = referenceOverlay.getContext("2d");
    if (!ctx) return;

    const rect = modalTile.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    const dpr = window.devicePixelRatio || 1;

    referenceOverlay.width = Math.round(width * dpr);
    referenceOverlay.height = Math.round(height * dpr);
    referenceOverlay.style.width = `${width}px`;
    referenceOverlay.style.height = `${height}px`;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const boardDiameter = 451.0;
    const ringRadii = [170.0, 162.0, 107.0, 99.0, 15.9, 6.35];
    const lineInner = 15.9;
    const lineOuter = 170.0;
    const rotationRad = -(9 * Math.PI) / 180;
    const canonicalAngles = Array.from({ length: 10 }, (_, i) => (i * Math.PI) / 10);

    const size = Math.min(width, height) - 1;
    if (size <= 0) return;

    const scale = size / boardDiameter;
    const cx = (width - 1) * 0.5;
    const cy = (height - 1) * 0.5;

    ctx.strokeStyle = "rgba(0, 229, 255, 0.92)";
    ctx.lineWidth = Math.max(1.2, size / 600);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (const radiusMm of ringRadii) {
      const radiusPx = radiusMm * scale;
      if (radiusPx <= 0) continue;
      ctx.beginPath();
      ctx.arc(cx, cy, radiusPx, 0, Math.PI * 2);
      ctx.stroke();
    }

    const innerPx = lineInner * scale;
    const outerPx = lineOuter * scale;
    for (const angle of canonicalAngles) {
      const theta = angle + rotationRad;
      const dx = Math.cos(theta);
      const dy = Math.sin(theta);

      ctx.beginPath();
      ctx.moveTo(cx + dx * innerPx, cy + dy * innerPx);
      ctx.lineTo(cx + dx * outerPx, cy + dy * outerPx);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(cx - dx * innerPx, cy - dy * innerPx);
      ctx.lineTo(cx - dx * outerPx, cy - dy * outerPx);
      ctx.stroke();
    }
  };

  const openModal = (camId) => {
    currentCam = camId;

    modalLabel.textContent = camId.toUpperCase();
    modalTitle.textContent = `Calibration - ${camId.toUpperCase()}`;

    modalTile.classList.remove("is-live");
    modalImg.onload = () => modalTile.classList.add("is-live");
    modalImg.onerror = () => modalTile.classList.remove("is-live");
    modalImg.src = `/api/stream/${camId}?t=${Date.now()}`;

    modal.classList.add("is-open");
    modal.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
    scheduleOverlayDraw();
  };

  const closeModal = () => {
    currentCam = null;
    modal.classList.remove("is-open");
    modal.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
    modalImg.removeAttribute("src");
    modalTile.classList.remove("is-live");
  };

  document.addEventListener("click", async (e) => {
    const openBtn = e.target.closest("[data-open-calibration]");
    if (openBtn) return openModal(openBtn.dataset.openCalibration);

    if (e.target.closest("[data-close-modal]")) return closeModal();

    const actionBtn = e.target.closest("[data-modal-action]");
    if (actionBtn && currentCam) {
      const action = actionBtn.dataset.modalAction;
      actionBtn.disabled = true;
      try {
        await callBackend(currentCam, action);
        modalImg.src = `/api/stream/${currentCam}?t=${Date.now()}`;
        const preview = document.getElementById(currentCam);
        if (preview) preview.src = `/api/stream/${currentCam}?t=${Date.now()}`;
      } finally {
        actionBtn.disabled = false;
      }
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.classList.contains("is-open")) closeModal();
  });

  if (referenceOverlay && "ResizeObserver" in window) {
    const observer = new ResizeObserver(() => {
      if (modal.classList.contains("is-open")) scheduleOverlayDraw();
    });
    observer.observe(modalTile);
  } else {
    window.addEventListener("resize", () => {
      if (modal.classList.contains("is-open")) scheduleOverlayDraw();
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
