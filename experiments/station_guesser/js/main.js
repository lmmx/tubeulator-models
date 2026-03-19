const Main = (() => {
  let stationsLoaded = false;

  function init() {
    document.getElementById("stationsIn").addEventListener("change", e => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = ev => {
        try {
          const data = JSON.parse(ev.target.result);
          if (Array.isArray(data) && data.length > 0 && data[0].name) {
            State.loadStations(data);
            stationsLoaded = true;
            updateLoadStatus();
          }
        } catch (err) { alert("Bad stations JSON: " + err.message); }
      };
      reader.readAsText(file);
    });

    document.getElementById("rulingsIn").addEventListener("change", e => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = ev => {
        try {
          State.loadRulings(JSON.parse(ev.target.result));
          updateLoadStatus();
        } catch (err) { alert("Bad rulings JSON: " + err.message); }
      };
      reader.readAsText(file);
    });

    // keyboard shortcuts
    document.addEventListener("keydown", e => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (e.key === "n" || e.key === "N") Navigator.rule("ruled-out");
      if (e.key === "m" || e.key === "M") Navigator.rule("maybe");
      if (e.key === "s" || e.key === "S") Navigator.skip();
    });
  }

  function updateLoadStatus() {
    const el = document.getElementById("loadStatus");
    const c = State.counts();
    const parts = [];
    if (stationsLoaded) parts.push(c.total + " stations loaded");
    if (c.out > 0 || c.maybe > 0) {
      parts.push(c.out + " ruled out, " + c.maybe + " maybe");
      parts.push(c.unknown + " remaining to review");
    }
    el.textContent = parts.join(" · ");

    if (stationsLoaded && c.unknown > 0) {
      document.getElementById("startBtn").classList.remove("hidden");
    }
  }

  function start() {
    document.getElementById("loadScreen").classList.add("hidden");
    document.getElementById("gameScreen").classList.remove("hidden");

    Navigator.buildQueue("borough");
    Navigator.initMinimap();
    Navigator.showCurrent();
  }

  init();
  return { start };
})();