// Dashboard tiles, System panel and the demo-data switch.
import * as api from './api.js';
import { el, $, clear, fmtDistance, showMessage, withBusy, emit } from './ui.js';

let systemInfo = null;
export const getSystemInfo = () => systemInfo;

const tile = (num, label, sub) => el('div', { class: 'tile' },
  el('div', { class: 'num', text: String(num) }), el('div', { class: 'lbl', text: label }),
  sub ? el('div', { class: 'sub', text: sub }) : null);

function renderStats(s) {
  const by = s.sessions.by_status || {};
  const pot = s.potholes.by_gps_source || {};
  const demo = pot.synthetic || 0;
  const cov = s.coverage || {};
  const tiles = clear($('#stat-tiles'));
  tiles.append(
    tile(s.sessions.total, 'Sessions',
      `done ${by.done || 0}, running ${by.running || 0}, queued ${by.queued || 0}, failed ${by.failed || 0}`),
    tile(s.observations.total, 'Tracked pothole sightings',
      `${s.observations.with_position} with position, ${s.observations.without_position} without`),
    tile(s.potholes.total, 'Mapped potholes',
      `${pot.recorded || 0} recorded, ${demo} demo (synthetic GPS)`),
    tile(fmtDistance(cov.surveyed_m), 'Surveyed road length',
      `${cov.edges} road segments (recorded sessions only). All other road: not surveyed, unknown.`
      + (cov.demo_m > 0 ? `
Demo drive: ${fmtDistance(cov.demo_m)} (synthetic, not a survey)` : '')));
}

const kv = (rows) => el('dl', { class: 'kv' }, rows.flatMap(([k, v]) => [el('dt', { text: k }), el('dd', { text: String(v) })]));

function renderSystem(sys) {
  const box = clear($('#system-panel'));
  const m = sys.model || {};
  const g = sys.graph || {};
  const st = sys.settings || {};
  box.append(
    el('div', { class: 'sys-title', text: 'Model' }),
    kv([['Architecture', m.architecture || 'n/a'], ['Weights file', m.weights || 'n/a'],
      ['Weights present', m.exists ? 'yes' : 'no'],
      ['SHA-256', m.sha256 ? `${m.sha256.slice(0, 12)}...` : 'n/a'],
      ['Loaded', m.loaded ? 'yes' : 'not yet (loads on first job)'], ['Device', m.device || 'n/a']]),
    el('div', { class: 'sys-title', text: 'Road network' }),
    g.loaded
      ? kv([['Loaded', 'yes'], ['Nodes', g.n_nodes], ['Edges', g.n_edges], ['Attribution', g.attribution || '© OpenStreetMap contributors']])
      : kv([['Loaded', 'no (route planner unavailable)']]),
    el('div', { class: 'sys-title', text: 'Settings (provisional)' }),
    el('p', { class: 'hint', text: 'These thresholds are provisional and have not been validated.' }));
  for (const [group, values] of Object.entries(st)) {
    box.append(kv(Object.entries(values).map(([k, v]) => [`${group}.${k}`, v === null ? 'not set' : v])));
  }
  box.append(el('p', { class: 'hint', text: `Server version ${sys.version}` }));
}

function renderDemoState(sys) {
  const demo = (sys && sys.demo) || {};
  $('#demo-banner').hidden = !demo.active;
  $('#demo-none-hint').hidden = demo.sessions !== 0;
}

export async function refreshDashboard() {
  const err = $('#dash-error');
  try {
    const [stats, sys] = await Promise.all([api.getStats(), api.getSystem()]);
    systemInfo = sys;
    showMessage(err, null);
    renderStats(stats);
    renderSystem(sys);
    renderDemoState(sys);
    const box = $('#include-synthetic');
    if (box.getAttribute('aria-busy') !== 'true') box.checked = !!sys.include_synthetic;  // not while the user's toggle is pending
  } catch (e) {
    showMessage(err, `Could not load the dashboard: ${e.message}`);
  }
}

export function initDashboard() {
  const box = $('#include-synthetic');
  box.addEventListener('change', async () => {
    const msg = $('#rebuild-error');
    const wanted = box.checked;
    await withBusy(box, async () => {
      try {
        await api.rebuildMap(wanted);
        showMessage(msg, null);
        emit('data-changed');
      } catch (e) {
        box.checked = !wanted;
        showMessage(msg, `Map rebuild failed: ${e.message}`);
      }
    });
  });
}
