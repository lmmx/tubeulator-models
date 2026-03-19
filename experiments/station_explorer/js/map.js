const MapView = (() => {
  let map, markers = [], circle = null;
  let showLabels = true;

  function init() {
    map = L.map("map").setView([51.52, -0.08], 12);
    L.tileLayer("https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png", {
      attribution: '&copy; OpenStreetMap contributors, Tiles: HOT',
      maxZoom: 19
    }).addTo(map);
    const b = Filters.getBoroughCoords();
    L.circleMarker(b, {
      radius: 7, color: "#d85a30", fillColor: "#d85a30", fillOpacity: 1, weight: 2
    }).addTo(map).bindTooltip("Borough", { permanent: true, direction: "right", offset: [8, 0] });
  }

  function getMap() { return map; }

  function toggleLabels() {
    showLabels = !showLabels;
    document.getElementById("labelsBtn").classList.toggle("active", showLabels);
    render();
  }

  function svUrl(lat, lon) {
    return "https://www.google.com/maps?layer=c&cbll=" + lat + "," + lon;
  }

  function render() {
    markers.forEach(m => map.removeLayer(m));
    markers = [];
    if (circle) { map.removeLayer(circle); circle = null; }

    const r = Filters.getRadius();
    if (r > 0) {
      const b = Filters.getBoroughCoords();
      circle = L.circle(b, {
        radius: r * 1000, color: "#d85a30", weight: 1.5,
        fillOpacity: 0.05, dashArray: "6 4"
      }).addTo(map);
    }

    let shown = 0;
    State.getStations().forEach(st => {
      if (!Filters.passes(st)) return;
      shown++;
      const s = State.getStatus(st.id);
      const isSel = Select.isSelected(st.id);

      const col = isSel ? "#2d6be4"
        : s === "maybe" ? "#1d9e75"
        : s === "ruled-out" ? "#bbb"
        : "#7F77DD";
      const size = isSel ? 8 : s === "ruled-out" ? 4 : 6;
      const opacity = s === "ruled-out" && !isSel ? 0.25 : 0.85;

      const m = L.circleMarker([st.lat, st.lon], {
        radius: size, color: col, fillColor: col,
        fillOpacity: opacity, weight: isSel ? 3 : (s === "maybe" ? 2 : 0)
      }).addTo(map);

      m.stationId = st.id;

      m.on("click", (e) => {
        if (Select.isActive()) {
          L.DomEvent.stopPropagation(e);
          Select.handleMarkerClick(st.id);
        }
      });

      if (showLabels) {
        const labelClass = s === "ruled-out" && !isSel ? "station-label faded" : "station-label";
        m.bindTooltip(st.name, {
          permanent: true, direction: "right", offset: [8, -2],
          className: labelClass
        });
      }

      if (!Select.isActive()) {
        const noOn = s === "ruled-out" ? " on" : "";
        const maybeOn = s === "maybe" ? " on" : "";
        m.bindPopup(
          '<div class="popup-name">' + st.name + '</div>' +
          '<div class="popup-lines">' + st.lines.join(" &middot; ") + '</div>' +
          '<div class="popup-actions">' +
            '<a href="' + svUrl(st.lat, st.lon) + '" target="_blank">Street View</a>' +
            '<button class="btn-no' + noOn + '" onclick="State.setStatus(\'' + st.id + '\',\'ruled-out\')">Not it</button>' +
            '<button class="btn-maybe' + maybeOn + '" onclick="State.setStatus(\'' + st.id + '\',\'maybe\')">Maybe</button>' +
          '</div>'
        , { maxWidth: 260 });
      }

      markers.push(m);
    });

    const c = State.counts();
    document.getElementById("stats").innerHTML =
      '<span>Showing ' + shown + ' of ' + c.total + '</span>' +
      '<span>' + c.unknown + ' unknown</span>' +
      '<span style="color:#0a7">' + c.maybe + ' maybe</span>' +
      '<span style="color:#a33">' + c.out + ' ruled out</span>';
  }

  function getMarkers() { return markers; }

  return { init, getMap, render, getMarkers, toggleLabels };
})();