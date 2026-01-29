(() => {
  const getParam = (name) => {
    const params = new URLSearchParams(window.location.search);
    const value = params.get(name);
    return value ? value.trim() : "";
  };

  const setText = (id, value, fallback = "-") => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = value || fallback;
  };

  setText("game-param-mode", getParam("mode"));
  setText("game-param-checkin", getParam("checkIn"));
  setText("game-param-checkout", getParam("checkOut"));
  setText("game-param-players", getParam("players"));
})();
