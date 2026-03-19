const Select = (() => {
  let active = false;
  let selected = new Set();

  function toggle() {
    active = !active;
    const btn = document.getElementById("selectBtn");
    btn.classList.toggle("active", active);
    document.getElementById("selectStatus").classList.toggle("hidden", !active);
    document.getElementById("selectActions").classList.toggle("hidden", !active);
    if (!active) {
      clear();
      MapView.render();
    }
  }

  function isActive() { return active; }

  function handleMarkerClick(stationId) {
    if (!active) return false;
    if (selected.has(stationId)) selected.delete(stationId);
    else selected.add(stationId);
    updateCount();
    MapView.render();
    return true; // signal that we consumed the click
  }

  function isSelected(id) { return selected.has(id); }

  function apply(action) {
    if (selected.size === 0) return;
    State.bulkSetStatus([...selected], action);
    clear();
  }

  function clear() {
    selected.clear();
    updateCount();
  }

  function updateCount() {
    const el = document.getElementById("selectCount");
    el.textContent = selected.size + " selected";
  }

  return { toggle, isActive, handleMarkerClick, isSelected, apply, clear };
})();