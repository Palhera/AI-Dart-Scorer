const DATA_COLLECTION_KEY = "dataCollectionEnabled";

/* ----- Game Mode Selection ----- */
(() => {
  const triggers = Array.from(document.querySelectorAll(".mode-trigger"));
  if (triggers.length === 0) return;
  const optionPanels = Array.from(
    document.querySelectorAll(".mode-card .mode-options")
  );

  const dataCollectionTrigger = document.querySelector(
    '.mode-trigger[data-mode="data-collection"]'
  );
  if (dataCollectionTrigger) {
    const enabled = localStorage.getItem(DATA_COLLECTION_KEY) === "true";
    const dataCollectionItem = dataCollectionTrigger.closest(".mode-item");
    const dataCollectionCard = dataCollectionTrigger.closest(".mode-card");
    const dataCollectionPanel = dataCollectionCard?.querySelector(".mode-options");

    if (!enabled) {
      dataCollectionItem?.classList.add("is-hidden");
      dataCollectionTrigger.disabled = true;
      dataCollectionTrigger.setAttribute("aria-checked", "false");
      dataCollectionTrigger.setAttribute("aria-expanded", "false");
      dataCollectionPanel?.setAttribute("aria-hidden", "true");
    } else {
      dataCollectionItem?.classList.remove("is-hidden");
      dataCollectionTrigger.disabled = false;
    }
  }

  function updateOptionHeights() {
    optionPanels.forEach((panel) => {
      panel.style.setProperty(
        "--mode-options-height",
        `${panel.scrollHeight}px`
      );
    });
  }

  function setExpanded(trigger, expanded) {
    trigger.setAttribute("aria-expanded", expanded ? "true" : "false");

    const card = trigger.closest(".mode-card");
    if (!card) return;
    const panel = card.querySelector(".mode-options");
    if (!panel) return;
    panel.setAttribute("aria-hidden", expanded ? "false" : "true");
  }

  function selectMode(trigger) {
    if (trigger.disabled) return;

    triggers.forEach((t) => {
      t.setAttribute("aria-checked", "false");
      const card = t.closest(".mode-card");
      if (card) card.classList.remove("is-selected");
      setExpanded(t, false);
    });

    const activeCard = trigger.closest(".mode-card");
    if (activeCard) activeCard.classList.add("is-selected");
    trigger.setAttribute("aria-checked", "true");
    setExpanded(trigger, true);
    updateOptionHeights();

    const mode = trigger.dataset.mode;
  }

  const initial =
    triggers.find((t) => t.getAttribute("aria-checked") === "true") ||
    triggers.find((t) => !t.disabled) ||
    triggers[0];

  selectMode(initial);
  updateOptionHeights();

  triggers.forEach((trigger) => {
    trigger.addEventListener("click", () => selectMode(trigger));
    trigger.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectMode(trigger);
      }
    });
  });

  window.addEventListener("load", updateOptionHeights);
  window.addEventListener("resize", () => {
    window.requestAnimationFrame(updateOptionHeights);
  });
})();

/* ----- Number of Players Selection ----- */
(() => {
  const container = document.querySelector(".number-of-players-options");
  if (!container) return;

  const buttons = Array.from(
    container.querySelectorAll(".number-of-players-card")
  );
  if (buttons.length === 0) return;

  const frame = container.querySelector(".players-frame");

  function updateFrame(target) {
    if (!frame || !target) return;
    frame.style.setProperty("--player-frame-x", `${target.offsetLeft}px`);
    frame.style.setProperty("--player-frame-y", `${target.offsetTop}px`);
    frame.style.setProperty("--player-frame-w", `${target.offsetWidth}px`);
    frame.style.setProperty("--player-frame-h", `${target.offsetHeight}px`);
  }

  function selectPlayers(button) {
    buttons.forEach((b) => {
      b.classList.remove("is-selected");
      b.setAttribute("aria-checked", "false");
    });

    button.classList.add("is-selected");
    button.setAttribute("aria-checked", "true");
    updateFrame(button);

    const players = button.dataset.players;
  }

  const initial =
    buttons.find((b) => b.getAttribute("aria-checked") === "true") || buttons[0];

  selectPlayers(initial);

  buttons.forEach((button) => {
    button.addEventListener("click", () => selectPlayers(button));
    button.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectPlayers(button);
      }
    });
  });

  const updateSelectedFrame = () => {
    const selected =
      buttons.find((b) => b.classList.contains("is-selected")) || initial;
    updateFrame(selected);
  };

  window.addEventListener("load", () =>
    window.requestAnimationFrame(updateSelectedFrame)
  );
  window.addEventListener("resize", () => {
    window.requestAnimationFrame(updateSelectedFrame);
  });

  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(() => {
      window.requestAnimationFrame(updateSelectedFrame);
    });
    observer.observe(container);
  }
})();

/* ----- Start Game Storage ----- */
(() => {
  const startButton = document.querySelector(".start-game");
  const modeTriggers = document.querySelectorAll(".mode-trigger");
  const playerButtons = document.querySelectorAll(".number-of-players-card");
  if (!startButton || modeTriggers.length === 0 || playerButtons.length === 0) {
    return;
  }

  function getSelectedMode() {
    const trigger =
      document.querySelector('.mode-trigger[aria-checked="true"]') ||
      document.querySelector(".mode-trigger:not([disabled])");
    if (!trigger) {
      return {
        mode: null,
        modeLabel: null,
        checkOut: null,
        checkIn: null,
        dataPath: null,
      };
    }

    const card = trigger.closest(".mode-card");
    const checkOutInput = card?.querySelector(
      'input[name^="checkout-"]:checked'
    );
    const checkInInput = card?.querySelector('input[name^="checkin-"]:checked');
    const dataPathInput = card?.querySelector('input[name="data-save-path"]');
    const modeTitle = trigger
      .querySelector(".mode-title")
      ?.textContent?.trim();
    const dataPathValue = dataPathInput?.value?.trim();

    return {
      mode: trigger.dataset.mode || null,
      modeLabel:
        trigger.dataset.modeLabel || modeTitle || trigger.dataset.mode || null,
      checkOut: checkOutInput ? checkOutInput.value : null,
      checkIn: checkInInput ? checkInInput.value : null,
      dataPath: dataPathValue ? dataPathValue : null,
    };
  }

  function getSelectedPlayers() {
    const selected =
      document.querySelector(".number-of-players-card.is-selected") ||
      document.querySelector('.number-of-players-card[aria-checked="true"]');
    if (!selected) return null;
    return selected.dataset.players || selected.textContent?.trim() || null;
  }

  function saveSelection() {
    const modeData = getSelectedMode();
    const payload = {
      mode: modeData.mode,
      modeLabel: modeData.modeLabel,
      checkOut: modeData.checkOut,
      checkIn: modeData.checkIn,
      dataPath: modeData.dataPath,
      players: getSelectedPlayers(),
      savedAt: Date.now(),
    };

    localStorage.setItem("dartGameConfig", JSON.stringify(payload));
  }

  function updateStartLink() {
    const modeData = getSelectedMode();
    if (!modeData.mode) return;
    startButton.href =
      modeData.mode === "data-collection" ? "/data-collection" : "/game";
  }

  startButton.addEventListener("click", saveSelection);
  modeTriggers.forEach((trigger) => {
    trigger.addEventListener("click", updateStartLink);
    trigger.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        updateStartLink();
      }
    });
  });

  updateStartLink();
})();

/* ----- Game Page Summary ----- */
(() => {
  const summary = document.querySelector("[data-game-summary]");
  if (!summary) return;

  const modeEl = summary.querySelector("[data-summary-mode]");
  const playersEl = summary.querySelector("[data-summary-players]");
  const checkoutEl = summary.querySelector("[data-summary-checkout]");
  const checkinEl = summary.querySelector("[data-summary-checkin]");

  let data = null;
  try {
    const raw = localStorage.getItem("dartGameConfig");
    data = raw ? JSON.parse(raw) : null;
  } catch {
    data = null;
  }

  const ruleLabel = (value, suffix) => {
    if (!value) return "Not set";
    const map = { straight: "Straight", double: "Double", master: "Master" };
    const base = map[value] || value;
    return `${base} ${suffix}`;
  };

  if (modeEl) {
    const rawMode = data?.modeLabel || data?.mode;
    if (rawMode) {
      const text = `${rawMode}`.trim();
      modeEl.textContent = /^\d+$/.test(text) ? `${text} Darts` : text;
    } else {
      modeEl.textContent = "Not set";
    }
  }
  if (playersEl) {
    const players = data?.players;
    playersEl.textContent = players
      ? `${players} Player${players === "1" ? "" : "s"}`
      : "Not set";
  }
  if (checkoutEl) {
    checkoutEl.textContent = ruleLabel(data?.checkOut, "Out");
  }
  if (checkinEl) {
    checkinEl.textContent = ruleLabel(data?.checkIn, "In");
  }
})();

/* ----- Data Collection Page ----- */
(() => {
  const page = document.querySelector("[data-collection]");
  if (!page) return;

  const grid = page.querySelector("[data-camera-grid]");
  const statusEl = page.querySelector("[data-capture-status]");
  const dataPathEl = page.querySelector("[data-data-path]");
  const lastIdEl = page.querySelector("[data-last-id]");
  const captureButton = page.querySelector("[data-capture]");

  let dataPath = null;
  try {
    const raw = localStorage.getItem("dartGameConfig");
    const config = raw ? JSON.parse(raw) : null;
    dataPath = config?.dataPath || null;
  } catch {
    dataPath = null;
  }

  if (dataPathEl) {
    dataPathEl.textContent = dataPath || "Not set";
  }

  if (!dataPath) {
    if (statusEl) {
      statusEl.textContent = "Select a save folder before starting this mode.";
    }
    if (captureButton) captureButton.disabled = true;
  }

  function setStatus(text) {
    if (statusEl) statusEl.textContent = text;
  }

  async function loadCameras() {
    try {
      const response = await fetch("/cameras");
      if (!response.ok) {
        setStatus("Failed to load cameras.");
        return [];
      }
      const data = await response.json();
      return Array.isArray(data?.available) ? data.available : [];
    } catch {
      setStatus("Failed to load cameras.");
      return [];
    }
  }

  function renderCameras(cameraIds) {
    if (!grid) return;
    grid.innerHTML = "";

    if (!cameraIds || cameraIds.length === 0) {
      const empty = document.createElement("div");
      empty.className = "camera-empty";
      empty.textContent = "No cameras available.";
      grid.appendChild(empty);
      return;
    }

    cameraIds.forEach((camId) => {
      const card = document.createElement("div");
      card.className = "camera-card";

      const header = document.createElement("div");
      header.className = "camera-header";
      header.textContent = `Camera ${camId}`;

      const img = document.createElement("img");
      img.className = "camera-stream";
      img.alt = `Camera ${camId} stream`;
      img.src = `/camera/${camId}/stream`;

      card.appendChild(header);
      card.appendChild(img);
      grid.appendChild(card);
    });
  }

  async function captureSet() {
    if (!dataPath) return;
    if (captureButton) captureButton.disabled = true;
    setStatus("Capturing...");

    try {
      const response = await fetch("/data-collection/capture", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ save_path: dataPath }),
      });

      if (!response.ok) {
        setStatus("Capture failed.");
        return;
      }

      const data = await response.json();
      if (data?.id && lastIdEl) {
        lastIdEl.textContent = data.id;
      }
      setStatus("Saved.");
    } catch {
      setStatus("Capture failed.");
    } finally {
      if (captureButton) captureButton.disabled = !dataPath;
    }
  }

  if (captureButton) {
    captureButton.addEventListener("click", captureSet);
  }

  window.addEventListener("keydown", (event) => {
    if (event.code !== "Space") return;
    if (
      document.activeElement &&
      ["INPUT", "TEXTAREA", "BUTTON", "A"].includes(
        document.activeElement.tagName
      )
    ) {
      return;
    }
    event.preventDefault();
    if (captureButton && !captureButton.disabled) captureSet();
  });

  loadCameras().then(renderCameras);
})();

/* ----- Settings: Data Collection Toggle ----- */
(() => {
  const toggle = document.getElementById("toggleDataCollection");
  if (!toggle) return;

  const enabled = localStorage.getItem(DATA_COLLECTION_KEY) === "true";
  toggle.checked = enabled;

  toggle.addEventListener("change", () => {
    localStorage.setItem(
      DATA_COLLECTION_KEY,
      toggle.checked ? "true" : "false"
    );
  });
})();

/* ----- Calibration / Detection Page ----- */
(() => {
  const fileInput = document.getElementById("fileInput");
  const runDetection = document.getElementById("runDetection");
  const preview = document.getElementById("preview");
  const result = document.getElementById("result");
  const warpMatrix = document.getElementById("warpMatrix");

  // Not on the calibration page -> do nothing
  if (!fileInput || !runDetection) return;

  const state = { previewUrl: null, file: null };

  function hideImage(img) {
    if (!img) return;
    img.style.display = "none";
    img.removeAttribute("src");
  }

  function showImage(img, src) {
    if (!img) return;
    img.src = src;
    img.style.display = "block";
  }

  function clearPreviews() {
    hideImage(preview);
    hideImage(result);
    if (state.previewUrl) {
      URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = null;
    }
    if (warpMatrix) {
      warpMatrix.value = "Run detection to compute the matrix.";
    }
  }

  // Initial UI state
  runDetection.disabled = true;
  clearPreviews();

  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      state.file = null;
      runDetection.disabled = true;
      clearPreviews();
      return;
    }

    state.file = file;
    runDetection.disabled = false;

    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = URL.createObjectURL(file);

    showImage(preview, state.previewUrl);
    hideImage(result);
  });

  async function runDetectionRequest() {
    if (!state.file || runDetection.disabled) return;

    runDetection.disabled = true;

    const formData = new FormData();
    formData.append("file", state.file);

    try {
      const response = await fetch("/keypoints", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        hideImage(result);
        if (warpMatrix) warpMatrix.value = "Detection failed.";
        return;
      }

      const data = await response.json();
      if (!data?.image) {
        hideImage(result);
        if (warpMatrix) warpMatrix.value = "No matrix returned.";
        return;
      }

      showImage(result, `data:image/png;base64,${data.image}`);
      if (warpMatrix) {
        const matrix = data.total_warp_matrix;
        if (Array.isArray(matrix) && matrix.length === 3) {
          const lines = matrix.map((row) =>
            row
              .map((value) => {
                const num = Number(value);
                return Number.isFinite(num) ? num.toFixed(6) : "0.000000";
              })
              .join(" ")
          );
          warpMatrix.value = lines.join("\n");
        } else {
          warpMatrix.value = "No matrix returned.";
        }
      }
    } catch {
      hideImage(result);
      if (warpMatrix) warpMatrix.value = "Detection failed.";
    } finally {
      runDetection.disabled = false;
    }
  }

  runDetection.addEventListener("click", runDetectionRequest);

  // Optional: avoid leaking object URLs if user leaves the page
  window.addEventListener("beforeunload", () => {
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  });
})();
