// Detection report and gallery for one session.
import * as api from './api.js';
import * as mapView from './map.js';
import { el, $, clear, safeImage, openImage, fmtPct, fmtClock, sourceLabel, isDemo, showMessage,
  showTab, on, NO_POSITION_TEXT, DEMO_LABEL } from './ui.js';

/** The fields of the processing report worth showing, with plain labels. Missing fields are skipped. */
function scalars(report) {
  const r = report || {};
  const v = r.video || {};
  const g = r.gps || {};
  const d = r.detector || {};
  const has = (x) => x !== null && x !== undefined;
  const rows = [
    ['Video', has(v.width) && has(v.height) ? `${v.width} x ${v.height}, ${has(v.fps) ? Number(v.fps).toFixed(0) : '?'} fps, ${has(v.duration_s) ? Number(v.duration_s).toFixed(1) : '?'} s` : null],
    ['Frames analysed', has(r.frames_processed) ? `${r.frames_processed} of ${r.frames_total} (every ${r.frame_stride}${r.frame_stride === 1 ? 'st' : r.frame_stride === 2 ? 'nd' : r.frame_stride === 3 ? 'rd' : 'th'} frame)` : null],
    ['Frames that failed', has(r.n_frame_errors) && r.n_frame_errors > 0 ? String(r.n_frame_errors) : null],
    ['Frame-level detections', has(r.n_detections) ? String(r.n_detections) : null],
    ['Tracks', has(r.n_tracks) ? `${r.n_tracks} (${r.n_tracks_confirmed} confirmed, ${r.n_tracks_rejected_short} too short to count)` : null],
    ['GPS fixes', has(g.n_fixes) ? `${g.n_used} usable of ${g.n_fixes}` : null],
    ['Detector', has(d.conf) ? `confidence at least ${d.conf}, device ${d.device === 'cpu' ? 'CPU' : `GPU ${d.device}`}` : null],
    ['Processing time', has(r.elapsed_s) ? `${Number(r.elapsed_s).toFixed(1)} s` : null],
  ];
  return rows.filter(([, value]) => value !== null);
}

function summary(session, report, observations) {
  const withPos = observations.filter((o) => o.position).length;
  const box = el('div', { class: 'card report' },
    el('strong', { text: session.name }), ' ',
    el('span', { class: `chip ${isDemo(session.gps_source) ? 'demo' : ''}`, text: sourceLabel(session.gps_source) }),
    el('dl', { class: 'kv' },
      el('dt', { text: 'Tracked potholes' }), el('dd', { text: String(observations.length) }),
      el('dt', { text: 'With GPS position' }), el('dd', { text: String(withPos) }),
      el('dt', { text: 'Without GPS position' }), el('dd', { text: String(observations.length - withPos) }),
      ...(report ? scalars(report).flatMap(([k, v]) => [el('dt', { text: k }), el('dd', { text: v })]) : [])));
  if (!report) box.append(el('p', { class: 'hint', text: 'No finished processing report for this session yet.' }));
  const warnings = (report && report.warnings) || [];
  if (warnings.length) {
    box.append(el('div', { class: 'message warn', role: 'status' }, el('strong', { text: 'Warnings' }),
      el('ul', {}, warnings.map((w) => el('li', { text: typeof w === 'string' ? w : JSON.stringify(w) })))));
  }
  return box;
}

function card(o, gpsSource, sessionName) {
  const demo = isDemo(gpsSource);
  const holder = safeImage(o.image_url, `Pothole detection, track ${o.track_id}`);
  holder.addEventListener('click', () => openImage(o.image_url, `${sessionName}: track ${o.track_id}, ${fmtPct(o.confidence)}`));
  const body = el('div', { class: 'body' },
    el('span', { class: 'pct', text: fmtPct(o.confidence) }),
    el('span', { text: `At ${fmtClock(o.t_video_s)} in the video` }),
    el('span', { text: `Tracked over ${o.n_frames} frame${o.n_frames === 1 ? '' : 's'}` }),
    o.position
      ? el('span', { text: `${o.position.lat.toFixed(5)}, ${o.position.lon.toFixed(5)} (accuracy ${o.position.accuracy_m ?? 'n/a'} m)` })
      : el('span', { class: 'err-text', text: NO_POSITION_TEXT }));
  if (demo) body.append(el('span', { class: 'chip demo', text: DEMO_LABEL }));
  if (o.position) {
    const b = el('button', { type: 'button', class: 'small', text: 'Show on map' });
    b.addEventListener('click', () => mapView.focusPothole(o.position.lat, o.position.lon, o.pothole_id));
    body.append(b);
  }
  return el('div', { class: `det ${demo ? 'demo' : ''}` }, holder, body);
}

export async function showDetections(session) {
  showTab('detections');
  const body = clear($('#det-body'));
  const err = $('#det-error');
  showMessage(err, null);
  body.append(el('p', { class: 'empty', text: 'Loading detections...' }));
  try {
    const [full, det] = await Promise.all([api.getSession(session.id), api.getDetections(session.id)]);
    clear(body).append(summary(full, full.report, det.observations));
    if (!det.observations.length) {
      body.append(el('p', { class: 'empty', text: 'No potholes were confirmed in this session.' }));
      return;
    }
    body.append(el('div', { class: 'gallery' }, det.observations.map((o) => card(o, det.gps_source, full.name))));
  } catch (e) {
    clear(body);
    showMessage(err, `Could not load detections: ${e.message}`);
  }
}

export function initDetections() {
  on('view-detections', showDetections);
}
