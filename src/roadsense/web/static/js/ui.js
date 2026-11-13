// Small DOM helpers shared by all modules. API text always goes through textContent, never innerHTML.

export const bus = new EventTarget();
export const emit = (name, detail) => bus.dispatchEvent(new CustomEvent(name, { detail }));
export const on = (name, fn) => bus.addEventListener(name, (e) => fn(e.detail));

export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of children.flat()) {
    if (child === undefined || child === null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const clear = (node) => { node.replaceChildren(); return node; };

export function fmtDistance(m) {
  if (m === null || m === undefined || Number.isNaN(m)) return 'n/a';
  return m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`;
}
export function fmtDuration(s) {
  if (s === null || s === undefined) return 'n/a';
  if (s < 60) return `${Math.round(s)} s`;
  const mins = Math.round(s / 60);
  return mins < 60 ? `${mins} min` : `${Math.floor(mins / 60)} h ${mins % 60} min`;
}
export const fmtPct = (f, digits = 0) => (f === null || f === undefined ? 'n/a' : `${(f * 100).toFixed(digits)}%`);
export function fmtDate(iso) {
  if (!iso) return 'n/a';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString();
}
export function fmtClock(s) {
  if (s === null || s === undefined) return 'n/a';
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, '0')}`;
}
export const isDemo = (gpsSource) => gpsSource === 'synthetic' || gpsSource === 'mixed';
export const DEMO_LABEL = 'Demo data (synthetic GPS)';
export function sourceLabel(gpsSource) {
  if (isDemo(gpsSource)) return DEMO_LABEL;
  if (gpsSource === 'recorded') return 'Recorded GPS';
  return 'No GPS trace';
}
export const NO_POSITION_TEXT = 'No reliable GPS position: not placed on the map';

/** Run an async action with the given buttons disabled meanwhile. */
export async function withBusy(buttons, fn) {
  const list = [buttons].flat().filter(Boolean);
  list.forEach((b) => { b.disabled = true; b.setAttribute('aria-busy', 'true'); });
  try { return await fn(); } finally {
    list.forEach((b) => { b.disabled = false; b.removeAttribute('aria-busy'); });
  }
}

/** A null/undefined fraction gives an indeterminate bar (no aria-valuenow, no fixed width). */
export function progressBar(fraction, label) {
  const indeterminate = fraction === null || fraction === undefined || Number.isNaN(fraction);
  const pct = indeterminate ? 0 : Math.max(0, Math.min(1, fraction)) * 100;
  return el('div', { class: `progress${indeterminate ? ' indeterminate' : ''}`, role: 'progressbar',
    'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-valuenow': indeterminate ? null : Math.round(pct),
    'aria-label': label || 'Progress' },
  el('div', { class: 'progress-fill', style: indeterminate ? null : `width:${pct.toFixed(1)}%` }));
}

export function showMessage(container, message, kind = 'error') {
  clear(container);
  if (!message) { container.hidden = true; return; }
  container.hidden = false;
  container.className = `message ${kind}`;
  container.setAttribute('role', kind === 'error' ? 'alert' : 'status');
  container.append(message instanceof Node ? message : document.createTextNode(message));
}

/** Image that shows a placeholder when it fails to load. */
export function safeImage(url, alt, cls = '') {
  const holder = el('div', { class: `img-holder ${cls}` });
  const img = el('img', { src: url, alt, loading: 'lazy' });
  img.addEventListener('error', () => {
    holder.classList.add('broken');
    img.remove();
    holder.append(el('span', { class: 'img-missing', text: 'Image unavailable' }));
  });
  holder.append(img);
  return holder;
}

export function openImage(url, caption) {
  const dlg = $('#image-dialog');
  clear($('#image-dialog-body')).append(safeImage(url, caption || 'Detection image', 'large'));
  $('#image-dialog-caption').textContent = caption || '';
  if (!dlg.open) dlg.showModal();
}

/** In-page confirmation dialog; resolves true when confirmed. */
export function confirmDialog(title, text, confirmLabel = 'Confirm') {
  const dlg = $('#confirm-dialog');
  $('#confirm-title').textContent = title;
  $('#confirm-text').textContent = text;
  $('#confirm-ok').textContent = confirmLabel;
  return new Promise((resolve) => {
    dlg.returnValue = 'cancel';
    dlg.addEventListener('close', () => resolve(dlg.returnValue === 'ok'), { once: true });
    dlg.showModal();
    $('#confirm-cancel').focus();
  });
}

export function showTab(name) {
  document.querySelectorAll('[role="tab"]').forEach((t) => {
    const active = t.dataset.tab === name;
    t.setAttribute('aria-selected', String(active));
    t.tabIndex = active ? 0 : -1;
  });
  document.querySelectorAll('.tab-panel').forEach((p) => { p.hidden = p.id !== `tab-${name}`; });
  emit('tab', name);
}
