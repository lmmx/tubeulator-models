(function () {
  MapView.init();
  Filters.bind();

  // state changes re-render map
  State.onUpdate(MapView.render);

  // file loading
  document.getElementById("fileIn").addEventListener("change", e => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      try {
        const data = JSON.parse(ev.target.result);
        if (Array.isArray(data) && data.length > 0 && data[0].name) {
          State.loadStations(data);
          document.getElementById("search").disabled = false;
          document.getElementById("statusFilter").disabled = false;
          document.getElementById("radiusFilter").disabled = false;
          document.getElementById("exportBtn").classList.remove("hidden");
          document.getElementById("importBtn").classList.remove("hidden");
          document.getElementById("selectBtn").classList.remove("hidden");
          document.getElementById("selectCount").classList.remove("hidden");
          document.getElementById("labelsBtn").classList.remove("hidden");
          Filters.buildPills();
        }
      } catch (err) { alert("Failed to parse: " + err.message); }
    };
    reader.readAsText(file);
  });

  document.getElementById("rulingsIn").addEventListener("change", e => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      try { State.importRulings(JSON.parse(ev.target.result)); }
      catch (err) { alert("Bad rulings JSON: " + err.message); }
    };
    reader.readAsText(file);
  });
})();
