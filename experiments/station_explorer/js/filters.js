const Filters = (() => {
  let query = "";
  let statusFilter = "all";
  let radius = 0;
  let lineFilter = new Set();

  const BOROUGH = [51.5013, -0.0931];

  function haversine(lat1, lon1, lat2, lon2) {
    const R = 6371, dLat = (lat2 - lat1) * Math.PI / 180,
          dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2 +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  function passes(st) {
    if (query && !st.name.toLowerCase().includes(query)) return false;
    const s = State.getStatus(st.id);
    if (statusFilter === "unknown" && s !== "unknown") return false;
    if (statusFilter === "maybe" && s !== "maybe") return false;
    if (statusFilter === "ruled-out" && s !== "ruled-out") return false;
    if (radius > 0 && haversine(BOROUGH[0], BOROUGH[1], st.lat, st.lon) > radius) return false;
    if (lineFilter.size > 0 && !st.lines.some(l => lineFilter.has(l))) return false;
    return true;
  }

  function getRadius() { return radius; }
  function getBoroughCoords() { return BOROUGH; }
  function getLineFilter() { return lineFilter; }

  function buildPills() {
    const lines = new Set();
    State.getStations().forEach(st => st.lines.forEach(l => lines.add(l)));
    const sorted = [...lines].sort();
    const el = document.getElementById("linePills");
    el.innerHTML = sorted.map(l =>
      '<button class="pill" data-line="' + l + '">' + l + '</button>'
    ).join("");
    el.querySelectorAll(".pill").forEach(btn => {
      btn.addEventListener("click", () => {
        const l = btn.dataset.line;
        if (lineFilter.has(l)) { lineFilter.delete(l); btn.classList.remove("on"); }
        else { lineFilter.add(l); btn.classList.add("on"); }
        State.onUpdate.length; // triggers render via notify
        // need to manually trigger since filter change isn't a state change
        MapView.render();
      });
    });
  }

  function bind() {
    document.getElementById("search").addEventListener("input", e => {
      query = e.target.value.toLowerCase(); MapView.render();
    });
    document.getElementById("statusFilter").addEventListener("change", e => {
      statusFilter = e.target.value; MapView.render();
    });
    document.getElementById("radiusFilter").addEventListener("change", e => {
      radius = parseFloat(e.target.value); MapView.render();
    });
  }

  return { passes, getRadius, getBoroughCoords, getLineFilter, buildPills, bind };
})();