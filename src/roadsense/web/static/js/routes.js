// Route planner: endpoint selection, POST /api/routes, result cards and map highlighting.
import * as api from './api.js';
import * as mapView from './map.js';
import { el, $, clear, fmtDistance, fmtDuration, fmtPct, showMessage, withBusy, on, DEMO_LABEL } from './ui.js';

const COLORS = ['#2563eb', '#d97706', '#0d9488'];
const NO_GRAPH_TEXT = 'No road network is loaded. Build one with `python -m roadsense.routing.graph` (see README) and restart the server.';
const ENDPOINT = { origin: { input: '#r-origin', pick: '#pick-origin', name: 'origin' },
  destination: { input: '#r-dest', pick: '#pick-dest', name: 'destination' } };
const coords = { origin: null, destination: null };
let cards = [];
let demoExample = null;
const DEMO_WARNING = 'Demo mode is on: synthetic demonstration data is in the map used for this plan, so these results do not describe real roads, whichever roads the routes use.';
const EVIDENCE_TEXT = { adequate: 'Survey coverage: adequate', partial: 'Survey coverage: partial',
  insufficient: 'Survey coverage: too low to judge' };

function parseLatLon(text) {
  const m = /^\s*(-?\d+(?:\.\d+)?)\s*[,;\s]\s*(-?\d+(?:\.\d+)?)\s*$/.exec(text);
  if (!m) return null;
  const lat = Number(m[1]); const lon = Number(m[2]);
  return Math.abs(lat) <= 90 && Math.abs(lon) <= 180 ? { lat, lon } : null;
}

function setPoint(kind, latlng, { move = true } = {}) {
  const cfg = ENDPOINT[kind];
  coords[kind] = latlng ? { lat: latlng.lat, lon: latlng.lng ?? latlng.lon } : null;
  if (latlng && move) {
    $(cfg.input).value = `${coords[kind].lat.toFixed(6)}, ${coords[kind].lon.toFixed(6)}`;
    mapView.setEndpoint(kind, [coords[kind].lat, coords[kind].lon], (ll) => setPoint(kind, ll));
  }
  $(cfg.pick).setAttribute('aria-pressed', 'false');
}

function resetPickButtons() {
  for (const cfg of Object.values(ENDPOINT)) $(cfg.pick).setAttribute('aria-pressed', 'false');
}

function metric(label, value) { return [el('dt', { text: label }), el('dd', { text: value })]; }

function routeCard(r, i, onSelect) {
  const est = (v) => (v === null || v === undefined ? 'not estimated' : `about ${Number(v).toFixed(1)} (estimate)`);
  const hasDemo = r.demo_m > 0 || r.n_potholes_demo > 0;
  const labels = r.labels && r.labels.length ? r.labels : [`Route ${i + 1}`];
  const card = el('article', { class: `card route-card ${r.recommended ? 'rec' : ''}`, tabindex: '0',
    style: `--c:${COLORS[i % COLORS.length]}`, 'aria-label': `Route: ${labels.join(', ')}` },
  el('h4', {}, labels.join(' / '),
    r.recommended ? el('span', { class: 'chip best', text: 'Matches your setting' }) : null,
    EVIDENCE_TEXT[r.evidence] ? el('span', { class: `chip ev-${r.evidence}`, text: EVIDENCE_TEXT[r.evidence] }) : null),
  el('dl', { class: 'kv' },
    metric('Distance', fmtDistance(r.distance_m)),
    metric('Estimated time', fmtDuration(r.duration_s)),
    metric('Mapped potholes passed', String(r.n_potholes_observed)),
    metric('Surveyed share of route', fmtPct(r.surveyed_fraction)),
    metric('Assumed potholes, unsurveyed part', est(r.expected_potholes_unsurveyed)),
    metric('Estimated total (mapped + assumed)', est(r.expected_potholes_total)),
    hasDemo ? metric('From demo data', `${fmtDistance(r.demo_m || 0)}, ${r.n_potholes_demo || 0} pothole${r.n_potholes_demo === 1 ? '' : 's'} (synthetic)`) : []),
  r.explanation ? el('p', { text: r.explanation }) : null,
  el('p', { class: 'hint', text: 'Unsurveyed road is unknown, not pothole-free. Assumed and estimated counts rest on an assumed rate, not on observation.' }));
  const select = (active) => mapView.highlightRoute(active ? i : null);
  card.addEventListener('mouseenter', () => select(true));
  card.addEventListener('mouseleave', () => { if (!card.classList.contains('active')) select(false); });
  card.addEventListener('focus', () => select(true));
  card.addEventListener('click', () => onSelect(i));
  card.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(i); } });
  return card;
}

function costModelText(cm) {
  const items = [];
  const detour = cm && cm.max_detour_per_pothole_m;
  if (detour != null) items.push(`The more you slide towards "Avoid potholes", the further a route may detour to miss one mapped pothole: up to ${detour} m per pothole at full avoidance (provisional).`);
  const rate = cm && cm.prior_rate_per_100m;
  if (rate != null) {
    const source = cm.prior_source ? ` Source: ${cm.prior_source}.` : '';
    items.push(Number(rate) > 0
      ? `For unsurveyed road the planner assumes ${Number(rate).toFixed(2)} potholes per 100 m. This is an assumed rate, not an observation of that road.${source}`
      : `No pothole rate is available for unsurveyed road yet, so it adds no estimated potholes. It is unknown, not pothole-free.${source}`);
  }
  if (cm && cm.description) items.push(String(cm.description));
  if (cm && cm.surveyed_total_m != null && cm.mapped_potholes_total != null) {
    items.push(cm.includes_demo_data === true
      ? `Based on ${cm.mapped_potholes_total} mapped potholes over ${fmtDistance(Number(cm.surveyed_total_m))} of road, including synthetic demonstration data.`
      : `Based on ${cm.mapped_potholes_total} mapped potholes over ${fmtDistance(Number(cm.surveyed_total_m))} of surveyed road.`);
  }
  const known = ['max_detour_per_pothole_m', 'prior_rate_per_100m', 'prior_source', 'description', 'surveyed_total_m', 'mapped_potholes_total', 'includes_demo_data'];
  for (const [k, v] of Object.entries(cm || {})) if (!known.includes(k)) items.push(`${k.replace(/_/g, ' ')}: ${typeof v === 'object' ? JSON.stringify(v) : v}`);
  return items;
}

function renderResults(plan) {
  const box = clear($('#route-results'));
  const routes = plan.routes || [];
  const select = (i) => {
    cards.forEach((c, j) => c.classList.toggle('active', i === j));
    mapView.highlightRoute(i);
  };
  cards = routes.map((r, i) => routeCard(r, i, select));
  if (plan.demo_mode) {
    box.append(el('div', { class: 'message demo-warn', role: 'status' },
      el('strong', { text: `${DEMO_LABEL} in use. ` }), String((plan.caveats || [])[0] || DEMO_WARNING)));
  }
  const notes = plan.confidence_notes || [];
  if (notes.length) {
    box.append(el('div', { class: 'message warn', role: 'status' }, el('strong', { text: 'Confidence' }),
      el('ul', {}, notes.map((n) => el('li', { text: String(n) })))));
  }
  box.append(el('h3', { text: 'Routes' }), ...cards);
  mapView.drawRoutes(routes, routes.map((_, i) => COLORS[i % COLORS.length]));
  box.append(el('div', { class: 'card cost-model' }, el('strong', { text: 'How routes are scored' }),
    el('ul', {}, costModelText(plan.cost_model).map((t) => el('li', { text: t })))));
  const caveats = (plan.caveats || []).slice(plan.demo_mode ? 1 : 0);
  if (caveats.length) {
    box.append(el('div', { class: 'message info' }, el('strong', { text: 'Caveats' }),
      el('ul', {}, caveats.map((c) => el('li', { text: String(c) })))));
  }
  box.append(el('p', { class: 'hint', text: 'Solid sections of a route were driven by a processed recorded video. Dash-dot sections run along a synthetic demo drive (not a real survey). Dotted sections are unsurveyed (unknown, not pothole-free).' }));
}

export async function refreshRoutesInfo() {
  const unavailable = $('#route-unavailable');
  let available = false;
  try {
    const info = await api.getRoutesInfo();
    available = !!info.available;
    showMessage(unavailable, available ? null : NO_GRAPH_TEXT);
  } catch (e) {
    showMessage(unavailable, `Could not check the road network: ${e.message}`);
  }
  $('#route-form').querySelectorAll('input, button').forEach((n) => {
    if (n.getAttribute('aria-busy') !== 'true') n.disabled = !available;  // a request in flight owns its button
  });
  return available;
}

export function initRoutes() {
  const msg = $('#route-message');
  const avoid = $('#r-avoid');
  const out = $('#r-avoid-out');
  avoid.addEventListener('input', () => { out.textContent = `${Math.round(avoid.value * 100)}%`; });

  for (const kind of Object.keys(ENDPOINT)) {
    const cfg = ENDPOINT[kind];
    $(cfg.pick).addEventListener('click', () => {
      if ($(cfg.pick).getAttribute('aria-pressed') === 'true') { mapView.cancelPick(); resetPickButtons(); return; }
      resetPickButtons();
      $(cfg.pick).setAttribute('aria-pressed', 'true');
      mapView.armPick(kind, (ll) => setPoint(kind, ll), `Click the map to set the ${cfg.name}`);
    });
    $(cfg.input).addEventListener('change', () => {
      const text = $(cfg.input).value.trim();
      if (!text) { setPoint(kind, null); mapView.setEndpoint(kind, null); return; }
      const p = parseLatLon(text);
      if (!p) { showMessage(msg, `The ${cfg.name} must look like "12.9716, 77.5946" (latitude, longitude).`); return; }
      showMessage(msg, null);
      setPoint(kind, { lat: p.lat, lng: p.lon });
    });
  }

  // Optional example prepared by scripts/make_synthetic_demo.py. It is demonstration data and only
  // valid while the demo session is on the map, so re-check whenever data changes.
  const example = $('#route-example');
  example.addEventListener('click', () => {
    const ex = demoExample;
    if (!ex) return;
    setPoint('origin', { lat: ex.origin.lat, lng: ex.origin.lon });
    setPoint('destination', { lat: ex.destination.lat, lng: ex.destination.lon });
    mapView.fitTo([[ex.origin.lat, ex.origin.lon], [ex.destination.lat, ex.destination.lon]]);
    const note = ex.description ? ` ${ex.description}` : '';
    showMessage(msg, `${DEMO_LABEL}: example end points loaded.${note} Press "Find routes".`, 'info');
  });
  const checkExample = async () => {
    try {
      const ex = await api.getDemoRouteExample();
      demoExample = ex && ex.origin && ex.destination ? ex : null;
    } catch { demoExample = null; }
    example.hidden = !demoExample;
  };
  checkExample();
  on('data-changed', checkExample);

  $('#route-form').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    for (const [kind, cfg] of Object.entries(ENDPOINT)) {
      const p = parseLatLon($(cfg.input).value);
      if (!p) return showMessage(msg, `Set the ${cfg.name} by clicking the map or typing "lat, lon".`);
      coords[kind] = p;
    }
    showMessage(msg, null);
    const btn = $('#route-submit');
    await withBusy(btn, async () => {
      btn.textContent = 'Finding routes...';
      try {
        const plan = await api.planRoutes(coords.origin, coords.destination, Number(avoid.value));
        renderResults(plan);
      } catch (e) {
        clear($('#route-results'));
        mapView.clearRoutes();
        const text = { outside_graph: 'A point is too far from any road in the loaded network. Move it closer to a road.',
          no_route: 'No route exists between these two points on the loaded road network.' }[e.code];
        showMessage(msg, text ? `${text} (${e.message})` : e.message);
      } finally { btn.textContent = 'Find routes'; }
    });
  });
}
