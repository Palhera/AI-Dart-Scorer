/* ----- Game Mode Selection ----- */
(() => {
  const triggers = Array.from(document.querySelectorAll(".mode-trigger"));
  if (triggers.length === 0) return;
  const optionPanels = Array.from(
    document.querySelectorAll(".mode-card .mode-options")
  );

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
    console.log("Selected mode:", mode);
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
    console.log("Selected players:", players);
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
      return { mode: null, checkOut: null, checkIn: null };
    }

    const card = trigger.closest(".mode-card");
    const checkOutInput = card?.querySelector(
      'input[name^="checkout-"]:checked'
    );
    const checkInInput = card?.querySelector('input[name^="checkin-"]:checked');

    return {
      mode: trigger.dataset.mode || null,
      checkOut: checkOutInput ? checkOutInput.value : null,
      checkIn: checkInInput ? checkInInput.value : null,
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
      checkOut: modeData.checkOut,
      checkIn: modeData.checkIn,
      players: getSelectedPlayers(),
      savedAt: Date.now(),
    };

    localStorage.setItem("dartGameConfig", JSON.stringify(payload));
  }

  startButton.addEventListener("click", saveSelection);
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
    modeEl.textContent = data?.mode ? `${data.mode} Darts` : "Not set";
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

/* ----- Calibration / Detection Page ----- */
(() => {
  const fileInput = document.getElementById("fileInput");
  const runDetection = document.getElementById("runDetection");
  const preview = document.getElementById("preview");
  const result = document.getElementById("result");

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
        return;
      }

      const data = await response.json();
      if (!data?.image) {
        hideImage(result);
        return;
      }

      showImage(result, `data:image/png;base64,${data.image}`);
    } catch {
      hideImage(result);
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
