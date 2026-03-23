const State = (() => {
  let stations = [];
  let statuses = {};

  function loadStations(data) {
    stations = data;
    console.log("[state] loaded " + data.length + " stations");
  }

  function getStations() { return stations; }

  function loadRulings(data) {
    const incoming = Object.keys(data).length;
    const beforeCount = Object.keys(statuses).length;

    Object.entries(data).forEach(([id, val]) => {
      if (!statuses[id]) {
        statuses[id] = val;
      }
    });

    const afterCount = Object.keys(statuses).length;
    const added = afterCount - beforeCount;

    console.log("[rulings] file had " + incoming + " entries");
    console.log("[rulings] before merge: " + beforeCount + ", after: " + afterCount + ", added: " + added);

    // verify against stations
    const stationIds = new Set(stations.map(s => s.id));
    let matched = 0, orphaned = 0;
    Object.keys(statuses).forEach(id => {
      if (stationIds.has(id)) matched++;
      else orphaned++;
    });
    console.log("[rulings] " + matched + " match a station, " + orphaned + " orphaned");

    // count what counts() will see
    let testOut = 0, testMaybe = 0;
    stations.forEach(st => {
      const s = statuses[st.id] || "unknown";
      if (s === "ruled-out") testOut++;
      if (s === "maybe") testMaybe++;
    });
    console.log("[rulings] counts preview: " + testOut + " ruled-out, " + testMaybe + " maybe, " + (stations.length - testOut - testMaybe) + " unknown");
  }

  function getStatus(id) { return statuses[id] || "unknown"; }

  function setStatus(id, val) {
    if (val === "unknown") delete statuses[id];
    else statuses[id] = val;
  }

  function unruledStations() {
    return stations.filter(st => getStatus(st.id) === "unknown");
  }

  function counts() {
    let maybe = 0, out = 0;
    stations.forEach(st => {
      const s = getStatus(st.id);
      if (s === "maybe") maybe++;
      if (s === "ruled-out") out++;
    });
    return {
      total: stations.length, maybe, out,
      unknown: stations.length - maybe - out
    };
  }

  function exportRulings() {
    console.log("[export] exporting " + Object.keys(statuses).length + " entries");
    const blob = new Blob([JSON.stringify(statuses, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "rulings.json"; a.click();
    URL.revokeObjectURL(url);
  }

  return { loadStations, getStations, loadRulings, getStatus, setStatus,
           unruledStations, counts, exportRulings };
})();
