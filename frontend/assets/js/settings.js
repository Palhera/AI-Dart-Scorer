(() => {
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

  if (!modal || !modalImg || !modalTile) return;

  let currentCam = null;

  const callBackend = (camId, action) =>
    fetch("/api/camera/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cam_id: camId, action }),
    });

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
})();
