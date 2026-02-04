(() => {
  const getParam = (name) => {
    // Read a query parameter from the URL and normalize it to a trimmed string.
    // Empty or missing params are returned as an empty string.
    const params = new URLSearchParams(window.location.search);
    const value = params.get(name);
    return value ? value.trim() : "";
  };

  const setText = (id, value, fallback = "-") => {
    // Safely write text content into a target element.
    // A fallback is used to keep the UI stable when params are missing.
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = value || fallback;
  };

  // Populate the game summary UI from URL parameters.
  // These params are attached when navigating from the game selection screen.
  setText("game-param-mode", getParam("mode"));
  setText("game-param-checkin", getParam("checkIn"));
  setText("game-param-checkout", getParam("checkOut"));
  setText("game-param-players", getParam("players"));
})();
