// Leaflet map: potholes, surveyed road, session track, route lines, endpoint markers.
import * as api from './api.js';
import { el, $, safeImage, openImage, fmtPct, fmtDate, sourceLabel, isDemo, emit } from './ui.js';

const L = window.L;
const RECORDED = { color: '#7f1d1d', fillColor: '#dc2626', fillOpacity: 0.85, weight: 2 };
const DEMO = { color: '#7c3aed', fill: false, weight: 3, dashArray: '4 3' };

let map;
let potholeLayer; let demoLayer; let coverageLayer; let trackLayer; let routeLayer;
const markers = new Map(); // pothole id -> { marker, sig }
let routeLines = [];
let demoLines = [];
let endpoints = { origin: null, destination: null };
let pickCb = null;
let reloadTimer = null;
let loadSeq = 0;
let ready = false;

export function initMap() {
  map = L.map('map', { zoomControl: true }).setView([20, 0], 2);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19, attribution: '© OpenStreetMap contributors',
  }).addTo(map);
  for (const [name, z] of [['coverage', 410], ['routes', 420], ['track', 430], ['potholes', 450]]) {
    map.createPane(name).style.zIndex = z;
  }
  coverageLayer = L.layerGroup().addTo(map);
  routeLayer = L.layerGroup().addTo(map);
  trackLayer = L.layerGroup().addTo(map);
  potholeLayer = L.layerGroup().addTo(map);
  demoLayer = L.layerGroup().addTo(map);

  map.on('moveend', () => {
    if (!ready) return;
    clearTimeout(reloadTimer);
    reloadTimer = setTimeout(() => refreshMap().catch(() => {}), 400);
  });
  map.on('click', (e) => {
    if (!pickCb) return;
    const cb = pickCb;
    pickCb = null;
    document.body.classList.remove('picking');
    $('#pick-hint').hidden = true;
    cb(e.latlng);
  });

  const toggles = { 'lyr-potholes': potholeLayer, 'lyr-demo': demoLayer, 'lyr-coverage': coverageLayer };
  for (const [id, layer] of Object.entries(toggles)) {
    $(`#${id}`).addEventListener('change', (e) => {
      if (e.target.checked) layer.addTo(map); else map.removeLayer(layer);
    });
  }
  L.control.scale({ position: 'bottomleft' }).addTo(map);
}

const viewBbox = () => {
  const b = map.getBounds();
  return [b.getSouth(), b.getWest(), b.getNorth(), b.getEast()];
};

/** Reload potholes and coverage for the current view (or everything when `all`). */
export async function refreshMap({ all = false } = {}) {
  const seq = ++loadSeq;
  const bbox = all ? undefined : viewBbox();
  const [potholes, coverage] = await Promise.all([api.getPotholes(bbox), api.getCoverage(bbox)]);
  if (seq !== loadSeq) return { potholes, coverage };
  renderPotholes(potholes);
  renderCoverage(coverage);
  return { potholes, coverage };
}

function sig(p) { return `${p.gps_source}|${p.n_observations}|${p.n_sessions}|${p.confidence}`; }

function renderPotholes({ potholes, note }) {
  const seen = new Set();
  for (const p of potholes) {
    seen.add(p.id);
    const old = markers.get(p.id);
    if (old && old.sig === sig(p)) continue;
    if (old) old.marker.remove();
    const demo = isDemo(p.gps_source);
    const marker = L.circleMarker([p.lat, p.lon], {
      pane: 'potholes', radius: 8, ...(demo ? DEMO : RECORDED),
    });
    marker.bindPopup(() => potholePopup(p, note), { maxWidth: 300 });
    marker.addTo(demo ? demoLayer : potholeLayer);
    markers.set(p.id, { marker, sig: sig(p) });
  }
  for (const [id, entry] of markers) {
    if (!seen.has(id)) { entry.marker.remove(); markers.delete(id); }
  }
  emit('potholes-loaded', potholes.length);
}

const DEMO_DRIVE_TEXT = 'Demo drive (synthetic GPS): not a real survey';

function renderCoverage({ stretches, demo_stretches: demoStretches }) {
  coverageLayer.clearLayers();
  for (const line of stretches || []) {
    L.polyline(line, { pane: 'coverage', color: '#0891b2', weight: 6, opacity: 0.8, lineCap: 'round' }).addTo(coverageLayer);
  }
  demoLines.forEach((l) => demoLayer.removeLayer(l));
  demoLines = [];
  for (const line of demoStretches || []) {
    const pl = L.polyline(line, { pane: 'coverage', color: '#7c3aed', weight: 5, opacity: 0.85, dashArray: '10 8', lineCap: 'butt' });
    pl.bindTooltip(DEMO_DRIVE_TEXT, { sticky: true });
    pl.bindPopup(DEMO_DRIVE_TEXT);
    pl.addTo(demoLayer);
    demoLines.push(pl);
  }
}

function potholePopup(p, note) {
  const box = el('div', { class: 'popup' },
    el('h4', { text: `Pothole #${p.id}` }),
    el('div', {}, el('span', { class: `chip ${isDemo(p.gps_source) ? 'demo' : 'rec'}`, text: sourceLabel(p.gps_source) })),
    el('dl', { class: 'kv' },
      el('dt', { text: 'Confidence' }), el('dd', { text: `${fmtPct(p.confidence)} (best detection)` }),
      el('dt', { text: 'Sightings' }), el('dd', { text: `${p.n_observations} in ${p.n_sessions} session${p.n_sessions === 1 ? '' : 's'}` }),
      el('dt', { text: 'First seen' }), el('dd', { text: fmtDate(p.first_seen_utc) }),
      el('dt', { text: 'Last seen' }), el('dd', { text: fmtDate(p.last_seen_utc) })),
    note ? el('p', { class: 'note', text: note }) : null);
  const thumbs = el('div', { class: 'thumbs' }, el('span', { class: 'hint', text: 'Loading sightings...' }));
  box.append(thumbs);
  api.getPothole(p.id).then((d) => {
    thumbs.replaceChildren();
    for (const o of d.observations || []) {
      const holder = safeImage(o.image_url, `Sighting in ${o.session_name || 'session'}`);
      holder.addEventListener('click', () => openImage(o.image_url, `${o.session_name || ''} (${fmtPct(o.confidence)})`));
      thumbs.append(holder);
    }
    if (!thumbs.children.length) thumbs.append(el('span', { class: 'hint', text: 'No sightings listed.' }));
  }).catch((e) => {
    thumbs.replaceChildren(el('span', { class: 'err-text', text: `Could not load sightings: ${e.message}` }));
  });
  return box;
}

/** First-load framing. Returns true if data existed. */
export async function initialView(systemBbox) {
  let result = null;
  try { result = await refreshMap({ all: true }); } finally { ready = true; }
  const bounds = L.latLngBounds([]);
  result?.potholes.potholes.forEach((p) => bounds.extend([p.lat, p.lon]));
  result?.coverage.stretches?.forEach((s) => s.forEach((pt) => bounds.extend(pt)));
  result?.coverage.demo_stretches?.forEach((s) => s.forEach((pt) => bounds.extend(pt)));
  if (bounds.isValid()) {
    map.fitBounds(bounds.pad(0.2), { maxZoom: 18 });
  } else if (systemBbox) {
    map.fitBounds([[systemBbox[0], systemBbox[1]], [systemBbox[2], systemBbox[3]]]);
  } else {
    map.setView([20, 0], 2);
    $('#map-hint').hidden = false;
  }
}

export function hideHint() { $('#map-hint').hidden = true; }

/** Fit the view to a list of [lat, lon] points. */
export function fitTo(latlngs) {
  const bounds = L.latLngBounds(latlngs);
  if (bounds.isValid()) map.fitBounds(bounds.pad(0.3), { maxZoom: 17 });
}

export function showTrack(points, gpsSource) {
  trackLayer.clearLayers();
  const latlngs = points.map((p) => [p[0], p[1]]);
  if (!latlngs.length) return false;
  const demo = isDemo(gpsSource);
  L.polyline(latlngs, { pane: 'track', color: demo ? '#7c3aed' : '#374151', weight: 3, dashArray: demo ? '6 6' : '2 6', lineCap: 'round' }).addTo(trackLayer);
  map.fitBounds(L.latLngBounds(latlngs).pad(0.15), { maxZoom: 19 });
  return true;
}
export const clearTrack = () => trackLayer.clearLayers();

export async function focusPothole(lat, lon, potholeId) {
  map.setView([lat, lon], Math.max(map.getZoom(), 18));
  try { await refreshMap(); } catch { return; }
  const entry = markers.get(potholeId);
  if (entry) entry.marker.openPopup();
}

// ---- Routes ----

const toLatLngs = (g) => {
  if (!g) return [];
  if (Array.isArray(g)) return g.map((p) => [p[0], p[1]]);
  if (Array.isArray(g.coordinates)) return g.coordinates.map((p) => [p[1], p[0]]); // GeoJSON is lon,lat
  return [];
};
const sectionPath = (s) => toLatLngs(s.coords || s.geometry || s.coordinates || s.points || s.latlngs || (Array.isArray(s) ? s : null));

/** surveyed: solid; demo: long dash-dot; unsurveyed: dotted. Falls back to the boolean `surveyed`. */
function sectionDash(s) {
  const status = s.status || (s.surveyed ? 'surveyed' : 'unsurveyed');
  if (status === 'surveyed') return null;
  return status === 'demo' ? '14 6 2 6' : '4 10';
}

export function drawRoutes(routes, colors) {
  clearRoutes();
  const order = routes.map((r, i) => i).sort((a, b) => Number(routes[a].recommended) - Number(routes[b].recommended));
  const all = L.latLngBounds([]);
  for (const i of order) {
    const r = routes[i];
    const rec = !!r.recommended;
    const base = { pane: 'routes', color: colors[i], weight: rec ? 8 : 5, opacity: 0.9, lineCap: 'butt' };
    const group = L.featureGroup();
    const sections = Array.isArray(r.sections) && r.sections.length ? r.sections : null;
    if (sections) {
      for (const s of sections) {
        const path = sectionPath(s);
        if (path.length) L.polyline(path, { ...base, dashArray: sectionDash(s) }).addTo(group);
      }
    }
    if (!group.getLayers().length) L.polyline(toLatLngs(r.geometry), { ...base, dashArray: '4 10' }).addTo(group);
    group.addTo(routeLayer);
    routeLines[i] = { group, rec, base };
    const b = group.getBounds();
    if (b.isValid()) all.extend(b);
  }
  if (all.isValid()) map.fitBounds(all.pad(0.15));
  highlightRoute(null);
}

export function highlightRoute(index) {
  routeLines.forEach((line, i) => {
    if (!line) return;
    const active = index === i;
    const dim = index !== null && !active;
    line.group.eachLayer((l) => l.setStyle({ opacity: dim ? 0.3 : 0.95, weight: line.base.weight + (active ? 3 : 0) }));
    if (active) line.group.bringToFront();
    else if (index === null && line.rec) line.group.bringToFront();
  });
}

export function clearRoutes() { routeLayer.clearLayers(); routeLines = []; }

export function armPick(kind, cb, message) {
  pickCb = (latlng) => cb(latlng);
  document.body.classList.add('picking');
  const hint = $('#pick-hint');
  hint.textContent = message;
  hint.hidden = false;
  map.getContainer().focus();
}
export function cancelPick() {
  pickCb = null;
  document.body.classList.remove('picking');
  $('#pick-hint').hidden = true;
}

/** Place or move a draggable endpoint marker. `latlng` null removes it. */
export function setEndpoint(kind, latlng, onDrag) {
  endpoints[kind]?.remove();
  endpoints[kind] = null;
  if (!latlng) return;
  const label = kind === 'origin' ? 'A' : 'B';
  const icon = L.divIcon({
    className: '', iconSize: [28, 28], iconAnchor: [14, 14],
    html: `<div style="width:28px;height:28px;border-radius:50%;background:${kind === 'origin' ? '#15803d' : '#1d4ed8'};color:#fff;font:700 14px system-ui;display:flex;align-items:center;justify-content:center;border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.5)">${label}</div>`,
  });
  const m = L.marker(latlng, { draggable: true, icon, zIndexOffset: 1000, keyboard: true,
    title: kind === 'origin' ? 'Origin (drag to move)' : 'Destination (drag to move)' }).addTo(map);
  m.on('dragend', () => onDrag(m.getLatLng()));
  endpoints[kind] = m;
}
