const State = (() => {
  let stations = [];
  let statuses = {};
  const listeners = [];

  function onUpdate(fn) { listeners.push(fn); }
  function notify() { listeners.forEach(fn => fn()); }

  function loadStations(data) {
    stations = data;
    notify();
  }

  function getStations() { return stations; }

  function getStatus(id) { return statuses[id] || "unknown"; }

  function setStatus(id, target) {
    const cur = getStatus(id);
    if (cur === target) delete statuses[id];
    else statuses[id] = target;
    notify();
  }

  function bulkSetStatus(ids, target) {
    ids.forEach(id => {
      if (target === "unknown") delete statuses[id];
      else statuses[id] = target;
    });
    notify();
  }

  function getStatuses() { return statuses; }

  function importRulings(data) {
    statuses = data;
    notify();
  }

  function exportRulings() {
    const blob = new Blob([JSON.stringify(statuses, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "rulings.json"; a.click();
    URL.revokeObjectURL(url);
  }

  function counts() {
    let maybe = 0, out = 0;
    stations.forEach(st => {
      const s = getStatus(st.id);
      if (s === "maybe") maybe++;
      if (s === "ruled-out") out++;
    });
    return { total: stations.length, maybe, out, unknown: stations.length - maybe - out };
  }

  return { onUpdate, loadStations, getStations, getStatus, setStatus,
           bulkSetStatus, getStatuses, importRulings, exportRulings, counts };
})();
