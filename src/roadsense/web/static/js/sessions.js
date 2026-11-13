// Session list, actions and job polling.
import * as api from './api.js';
import * as mapView from './map.js';
import { el, $, clear, fmtDate, fmtDuration, progressBar, sourceLabel, isDemo, showMessage,
  withBusy, confirmDialog, emit, on } from './ui.js';

const cards = new Map(); // session id -> { session, statusEl, root }
let pollTimer = null;
let polling = false;

const isActive = (s) => s.job && (s.job.status === 'queued' || s.job.status === 'running');

function renderStatus(entry) {
  const { session: s } = entry;
  const job = s.job || { status: 'queued' };
  const box = clear(entry.statusEl);
  const partial = job.status === 'done' && job.outcome === 'partial';
  const chip = el('span', { class: `chip ${partial ? 'partial' : job.status}`, text: partial ? 'done (partial)' : job.status });
  box.append(chip);
  if (partial) {
    box.append(el('p', { class: 'hint', text: 'Some frames could not be analysed; results may be incomplete. See the report.' }));
  }
  if (job.status === 'running') {
    const determinate = job.stage === 'detecting' && typeof job.progress === 'number';
    box.append(' ', el('span', { class: 'hint', text: determinate
      ? `${job.stage} ${Math.round(job.progress * 100)}%` : (job.stage || 'working') }),
    progressBar(determinate ? job.progress : null, `Processing ${s.name}`));
  } else if (job.status === 'queued') {
    box.append(' ', el('span', { class: 'hint', text: 'waiting for the processor' }));
  } else if (job.status === 'failed') {
    box.append(el('p', { class: 'err-text', text: job.error || 'Processing failed (no error text).' }));
  }
}

function buildCard(s) {
  const root = el('article', { class: 'card session' });
  const err = el('div', { hidden: true });
  const statusEl = el('div', { class: 'status' });
  const entry = { session: s, statusEl, root };
  const btn = (label, fn, cls = 'small') => {
    const b = el('button', { type: 'button', class: cls, text: label });
    b.addEventListener('click', () => withBusy(b, async () => {
      showMessage(err, null);
      try { await fn(b); } catch (e) { showMessage(err, e.message); }
    }));
    return b;
  };
  const done = s.job && s.job.status === 'done';
  const busy = isActive(s);

  const include = el('input', { type: 'checkbox' });
  include.checked = !!s.include_in_map;
  include.addEventListener('change', async () => {
    include.disabled = true;
    try {
      await api.patchSession(s.id, { include_in_map: include.checked });
      showMessage(err, null);
      emit('data-changed');
    } catch (e) { include.checked = !include.checked; showMessage(err, e.message); }
    include.disabled = false;
  });

  const gpsChip = el('span', { class: `chip ${isDemo(s.gps_source) ? 'demo' : ''}`, text: sourceLabel(s.gps_source) });
  root.append(
    el('header', {}, el('span', { class: 'name', text: s.name }), gpsChip),
    el('div', { class: 'meta' },
      `${s.role} | ${fmtDate(s.created_at)}`, s.duration_s ? ` | ${fmtDuration(s.duration_s)}` : '',
      s.device ? ` | ${s.device}` : '', el('br'),
      done ? `${s.n_observations} tracked potholes (${s.n_with_position} with a GPS position)` : 'no results yet'),
    statusEl,
    el('div', { class: 'actions' },
      btn('View detections', () => emit('view-detections', s), 'small primary'),
      btn('Show on map', async () => {
        const t = await api.getTrack(s.id);
        if (!mapView.showTrack(t.points || [], t.gps_source)) throw new Error('This session has no GPS track to show.');
      }),
      btn('Reprocess', async () => { await api.reprocessSession(s.id); await loadSessions(); }),
      el('label', { class: 'inc' }, include, 'Include in map'),
      btn('Delete', async () => {
        const ok = await confirmDialog('Delete session?',
          `"${s.name}", its uploaded files, detections and images will be permanently deleted.`, 'Delete');
        if (!ok) return;
        await api.deleteSession(s.id);
        await loadSessions();
        emit('data-changed');
      }, 'small')),
    err);
  if (busy) root.querySelectorAll('.actions button').forEach((b) => { if (b.textContent === 'Reprocess' || b.textContent === 'Delete') b.title = 'Not available while processing'; });
  renderStatus(entry);
  cards.set(s.id, entry);
  return root;
}

export async function loadSessions() {
  const list = $('#sessions-list');
  const err = $('#sessions-error');
  let data;
  try {
    data = await api.listSessions();
    showMessage(err, null);
  } catch (e) {
    showMessage(err, `Could not load sessions: ${e.message}`);
    return;
  }
  cards.clear();
  clear(list);
  if (!data.sessions.length) {
    list.append(el('p', { class: 'empty', text: 'No sessions yet. Use the Upload tab to add a video.' }));
  }
  for (const s of data.sessions) list.append(buildCard(s));
  schedulePolling(data.sessions.some(isActive));
}

function schedulePolling(active) {
  if (active && !pollTimer) pollTimer = setInterval(poll, 1000);
  if (!active && pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function poll() {
  if (polling) return;
  polling = true;
  try {
    let finished = false;
    for (const entry of cards.values()) {
      if (!isActive(entry.session)) continue;
      const st = await api.getSessionStatus(entry.session.id);
      entry.session.job = st.job;
      if (isActive(entry.session)) renderStatus(entry); else finished = true;
    }
    if (finished) {
      await loadSessions();
      emit('data-changed');
      emit('sessions-finished');
    }
  } catch { /* connection banner already handles failures; keep polling */ } finally { polling = false; }
}

export function initSessions() {
  on('sessions-changed', loadSessions);
}
