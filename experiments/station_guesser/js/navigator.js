const Navigator = (() => {
  let queue = [];
  let currentIdx = 0;
  let minimap = null;
  let currentMarker = null;
  let stationMarkers = [];

  const BOROUGH = [51.5013, -0.0931];

  function haversine(lat1, lon1, lat2, lon2) {
    const R = 6371, dLat = (lat2 - lat1) * Math.PI / 180,
          dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2 +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  function buildQueue(mode) {
    const unruled = State.unruledStations();
    if (mode === "random") {
      queue = unruled.slice();
      for (let i = queue.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [queue[i], queue[j]] = [queue[j], queue[i]];
      }
    } else {
      // nearest-neighbor from Borough
      queue = [];
      const remaining = new Set(unruled.map(s => s.id));
      const lookup = {};
      unruled.forEach(s => lookup[s.id] = s);
      let curLat = BOROUGH[0], curLon = BOROUGH[1];

      while (remaining.size > 0) {
        let bestId = null, bestDist = Infinity;
        remaining.forEach(id => {
          const s = lookup[id];
          const d = haversine(curLat, curLon, s.lat, s.lon);
          if (d < bestDist) { bestDist = d; bestId = id; }
        });
        remaining.delete(bestId);
        queue.push(lookup[bestId]);
        curLat = lookup[bestId].lat;
        curLon = lookup[bestId].lon;
      }
    }
    currentIdx = 0;
  }

  function initMinimap() {
    minimap = L.map("minimap", {
      zoomControl: false, attributionControl: false
    }).setView([51.52, -0.08], 12);

    L.tileLayer("https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png", {
      maxZoom: 18
    }).addTo(minimap);

    // show all stations as tiny dots
    State.getStations().forEach(st => {
      const col = State.getStatus(st.id) === "ruled-out" ? "#555"
        : State.getStatus(st.id) === "maybe" ? "#1d9e75" : "#666";
      const m = L.circleMarker([st.lat, st.lon], {
        radius: 3, color: col, fillColor: col, fillOpacity: 0.5, weight: 0
      }).addTo(minimap);
      stationMarkers.push({ id: st.id, marker: m });
    });

    // Borough marker
    L.circleMarker(BOROUGH, {
      radius: 5, color: "#d85a30", fillColor: "#d85a30", fillOpacity: 1, weight: 0
    }).addTo(minimap);
  }

  function current() {
    return queue[currentIdx] || null;
  }

  function showCurrent() {
    const st = current();
    if (!st) {
      showDone();
      return;
    }

    // update info
    document.getElementById("stationName").textContent = st.name;
    document.getElementById("stationLines").textContent = st.lines.join(" · ");

    // progress
    const c = State.counts();
    document.getElementById("progress").innerHTML =
      (currentIdx + 1) + " / " + queue.length + " remaining" +
      " &nbsp;·&nbsp; " + c.maybe + " maybe &nbsp;·&nbsp; " + c.out + " ruled out";

    // street view iframe
    const svUrl = "https://maps.google.com/maps?output=svembed&cbll=" +
      st.lat + "," + st.lon + "&cbp=12,0,,0,0&layer=c";
    document.getElementById("svFrame").src = svUrl;

    const directUrl = "https://www.google.com/maps?layer=c&cbll=" + st.lat + "," + st.lon;
    document.getElementById("svDirectLink").href = directUrl;

    // minimap
    if (minimap) {
      if (currentMarker) minimap.removeLayer(currentMarker);
      currentMarker = L.circleMarker([st.lat, st.lon], {
        radius: 8, color: "#2d6be4", fillColor: "#2d6be4",
        fillOpacity: 1, weight: 3
      }).addTo(minimap).bindTooltip(st.name, {
        permanent: true, direction: "right", offset: [10, 0],
        className: "current-label"
      });
      minimap.setView([st.lat, st.lon], 14);
    }

    // update minimap dot colors
    stationMarkers.forEach(sm => {
      const s = State.getStatus(sm.id);
      const col = sm.id === st.id ? "#2d6be4"
        : s === "ruled-out" ? "#555"
        : s === "maybe" ? "#1d9e75" : "#666";
      sm.marker.setStyle({ color: col, fillColor: col });
    });
  }

  function advance() {
    // rebuild remaining queue from current position using nearest-neighbor
    const mode = document.getElementById("startPoint").value;
    const st = current();
    if (!st) { showDone(); return; }

    // remove current from queue and rebuild
    const remaining = queue.filter((s, i) => i > currentIdx && State.getStatus(s.id) === "unknown");

    if (remaining.length === 0) {
      showDone();
      return;
    }

    if (mode === "borough") {
      // re-sort remaining by nearest to current station
      const sorted = [];
      const left = new Set(remaining.map(s => s.id));
      const lookup = {};
      remaining.forEach(s => lookup[s.id] = s);
      let curLat = st.lat, curLon = st.lon;

      while (left.size > 0) {
        let bestId = null, bestDist = Infinity;
        left.forEach(id => {
          const s = lookup[id];
          const d = haversine(curLat, curLon, s.lat, s.lon);
          if (d < bestDist) { bestDist = d; bestId = id; }
        });
        left.delete(bestId);
        sorted.push(lookup[bestId]);
        curLat = lookup[bestId].lat;
        curLon = lookup[bestId].lon;
      }

      queue = queue.slice(0, currentIdx + 1).concat(sorted);
    }

    currentIdx++;
    showCurrent();
  }

  function rule(status) {
    const st = current();
    if (!st) return;
    State.setStatus(st.id, status);
    advance();
  }

  function skip() {
    advance();
  }

  function showDone() {
    document.getElementById("gameScreen").classList.add("hidden");
    document.getElementById("doneScreen").classList.remove("hidden");
    const c = State.counts();
    document.getElementById("doneSummary").textContent =
      "You reviewed all remaining stations. " +
      c.maybe + " marked maybe, " + c.out + " ruled out, " +
      c.unknown + " still unknown.";
  }

  return { buildQueue, initMinimap, current, showCurrent, rule, skip };
})();
