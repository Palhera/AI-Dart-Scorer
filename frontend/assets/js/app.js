const startup = document.querySelector(".startup");
const main = document.querySelector("main");
const footer = document.querySelector("footer");
const hasStartupUI = Boolean(startup && main && footer);

let refreshGameModeHeights = null;
let timer = null;

/* ----- Startup Polling ----- */
async function startupPoll() {
  if (!hasStartupUI) return;
  try {
    const r = await fetch("/api/status", { cache: "no-store" });
    const data = await r.json();

    const ready = Boolean(data && data.ready);

    startup.hidden = ready;
    main.hidden = !ready;
    footer.hidden = !ready;

    if (ready) {
      if (typeof refreshGameModeHeights === "function") {
        window.requestAnimationFrame(refreshGameModeHeights);
      }
      if (timer) clearInterval(timer);
    }
  } catch {
    startup.hidden = false;
    main.hidden = true;
    footer.hidden = true;
  }
}

if (hasStartupUI) {
  startupPoll();
  timer = setInterval(startupPoll, 500);
}

/* ----- Data Collection Mode Visibility ----- */
(() => {
  const option = document.getElementById("game-mode-data-collection");
  if (!option) return;

  const key = "aiDartScorer.settings.showDataCollection";
  const show = localStorage.getItem(key) === "true";

  option.hidden = !show;
  option.setAttribute("aria-hidden", String(!show));
  option.setAttribute("aria-disabled", show ? "false" : "true");
})();

/* ----- Game Mode Selection ----- */
(() => {
  const options = Array.from(document.querySelectorAll(".game-mode-option"));
  if (options.length === 0) return;

  const isDisabled = (option) => option.getAttribute("aria-disabled") === "true";

  const configs = options
    .map((option) => option.querySelector(".game-mode-config"))
    .filter(Boolean);

  const updateConfigHeights = () => {
    for (const config of configs) {
      const styles = window.getComputedStyle(config);
      const borderTop = parseFloat(styles.borderTopWidth) || 0;
      const borderBottom = parseFloat(styles.borderBottomWidth) || 0;
      const safeHeight = config.scrollHeight + borderTop + borderBottom;

      config.style.setProperty("--game-mode-config-height", `${safeHeight}px`);
    }
  };

  refreshGameModeHeights = updateConfigHeights;

  const selectOption = (option) => {
    if (!option || isDisabled(option)) return;

    for (const item of options) item.classList.remove("is-selected");
    option.classList.add("is-selected");

    updateConfigHeights();
  };

  const initial =
    options.find((o) => o.classList.contains("is-selected") && !isDisabled(o)) ||
    options.find((o) => !isDisabled(o)) ||
    options[0];

  selectOption(initial);

  for (const option of options) {
    if (isDisabled(option)) continue;

    if (!option.hasAttribute("tabindex")) option.tabIndex = 0;

    option.addEventListener("click", () => selectOption(option));
    option.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectOption(option);
      }
    });
  }

  window.addEventListener("load", updateConfigHeights);
  window.addEventListener("resize", () => {
    window.requestAnimationFrame(updateConfigHeights);
  });

  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(() => {
      window.requestAnimationFrame(updateConfigHeights);
    });
    for (const config of configs) observer.observe(config);
  }
})();

/* Number Of Player Selection */
(() => {
  const container = document.querySelector(".player-count-options");
  const frame = container?.querySelector(".players-frame");
  const options = Array.from(document.querySelectorAll(".player-count-option"));
  if (!container || !frame || options.length === 0) return;

  const updateFrameTo = (option) => {
    if (!option) return;

    const cRect = container.getBoundingClientRect();
    const oRect = option.getBoundingClientRect();

    const x = oRect.left - cRect.left;
    const y = oRect.top - cRect.top;
    const w = oRect.width;
    const h = oRect.height;

    container.style.setProperty("--player-frame-x", `${x}px`);
    container.style.setProperty("--player-frame-y", `${y}px`);
    container.style.setProperty("--player-frame-w", `${w}px`);
    container.style.setProperty("--player-frame-h", `${h}px`);
  };

  const selectOption = (option) => {
    if (!option) return;

    for (const item of options) item.classList.remove("is-selected");
    option.classList.add("is-selected");

    window.requestAnimationFrame(() => updateFrameTo(option));
  };

  const getSelected = () =>
    options.find((o) => o.classList.contains("is-selected")) || options[0];

  selectOption(getSelected());

  for (const option of options) {
    if (!option.hasAttribute("tabindex")) option.tabIndex = 0;

    option.addEventListener("click", () => selectOption(option));
    option.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectOption(option);
      }
    });
  }

  const sync = () => window.requestAnimationFrame(() => updateFrameTo(getSelected()));

  window.addEventListener("load", sync);
  window.addEventListener("resize", sync);

  if ("ResizeObserver" in window) {
    const ro = new ResizeObserver(sync);
    ro.observe(container);
    for (const option of options) ro.observe(option);
  }
})();

/* ----- Attach Game Params To Start Link ----- */
(() => {
  const startLink = document.querySelector(".start-game");
  if (!startLink) return;

  const isDisabled = (option) => option?.getAttribute("aria-disabled") === "true";

  const getSelectedModeOption = () => {
    const options = Array.from(document.querySelectorAll(".game-mode-option"));
    if (options.length === 0) return null;

    return (
      options.find((option) => option.classList.contains("is-selected") && !isDisabled(option)) ||
      options.find((option) => !isDisabled(option)) ||
      options[0]
    );
  };

  const getCheckedValue = (container, namePrefix) => {
    if (!container) return null;
    const input = container.querySelector(`input[name^="${namePrefix}"]:checked`);
    return input ? input.value : null;
  };

  const getSelectedPlayers = () => {
    const options = Array.from(document.querySelectorAll(".player-count-option"));
    if (options.length === 0) return null;
    const selected = options.find((option) => option.classList.contains("is-selected")) || options[0];
    return selected?.getAttribute("value") || selected?.textContent?.trim() || null;
  };

  const updateStartLink = () => {
    const modeOption = getSelectedModeOption();
    if (!modeOption) return;

    const modeTitle = modeOption.querySelector(".game-mode-title")?.textContent?.trim() || "";
    const checkOut = getCheckedValue(modeOption, "checkout-");
    const checkIn = getCheckedValue(modeOption, "checkin-");
    const players = getSelectedPlayers();

    const params = new URLSearchParams();
    if (modeTitle) params.set("mode", modeTitle);
    if (checkIn) params.set("checkIn", checkIn);
    if (checkOut) params.set("checkOut", checkOut);
    if (players) params.set("players", players);

    const query = params.toString();
    startLink.setAttribute("href", query ? `/game?${query}` : "/game");
  };

  startLink.addEventListener("click", updateStartLink);
  startLink.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      updateStartLink();
    }
  });
})();
