// Wires the modules together.
import * as api from './api.js';
import * as mapView from './map.js';
import { initDashboard, refreshDashboard, getSystemInfo } from './dashboard.js';
import { initUpload } from './upload.js';
import { initSessions, loadSessions } from './sessions.js';
import { initDetections } from './detections.js';
import { initRoutes, refreshRoutesInfo } from './routes.js';
import { $, showTab, on } from './ui.js';

async function refreshData() {
  await Promise.all([
    refreshDashboard(),
    mapView.refreshMap().catch(() => {}),
    refreshRoutesInfo(),
  ]);
}

async function start() {
  mapView.initMap();
  initDashboard(); initUpload(); initSessions(); initDetections(); initRoutes();

  document.querySelectorAll('[role="tab"]').forEach((t) => t.addEventListener('click', () => showTab(t.dataset.tab)));
  $('nav[role="tablist"]').addEventListener('keydown', (e) => {
    const tabs = [...document.querySelectorAll('[role="tab"]')];
    const i = tabs.findIndex((t) => t.getAttribute('aria-selected') === 'true');
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1 }[e.key];
    if (next === undefined) return;
    const target = tabs[(next + tabs.length) % tabs.length];
    showTab(target.dataset.tab);
    target.focus();
  });

  let wasDown = false;
  api.onConnection((up) => {
    $('#conn-banner').hidden = up;
    if (up && wasDown) { wasDown = false; }
    if (!up) wasDown = true;
  });
  $('#conn-retry').addEventListener('click', async () => {
    const b = $('#conn-retry');
    b.disabled = true;
    try { await loadAll(); } finally { b.disabled = false; }
  });

  on('data-changed', () => { mapView.hideHint(); refreshData(); });
  on('tab', (name) => { if (name === 'sessions') loadSessions(); });

  await loadAll();
}

async function loadAll() {
  await Promise.all([refreshDashboard(), loadSessions(), refreshRoutesInfo()]);
  const sys = getSystemInfo();
  try {
    await mapView.initialView(sys && sys.graph && sys.graph.loaded ? sys.graph.bbox : null);
  } catch { /* banner/messages already shown by the API layer */ }
}

start();
