// Upload form for POST /api/sessions.
import * as api from './api.js';
import { $, clear, progressBar, showMessage, withBusy, showTab, emit } from './ui.js';

const ALLOWED = ['.mp4', '.mov', '.avi', '.mkv'];

export function initUpload() {
  const form = $('#upload-form');
  const submit = $('#up-submit');
  const msg = $('#up-message');

  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const video = $('#up-video').files[0];
    const gps = $('#up-gps').files[0];
    const sidecar = $('#up-sidecar').files[0];
    if (!video) return showMessage(msg, 'Choose a video file first.');
    const ext = video.name.slice(video.name.lastIndexOf('.')).toLowerCase();
    if (!ALLOWED.includes(ext)) {
      return showMessage(msg, `Unsupported video type "${ext}". Allowed: ${ALLOWED.join(', ')}.`);
    }
    if (gps && !gps.name.toLowerCase().endsWith('.csv')) return showMessage(msg, 'The GPS file must be a .csv file.');
    if (sidecar && !sidecar.name.toLowerCase().endsWith('.json')) return showMessage(msg, 'The sidecar must be a .json file.');
    if ($('#up-synthetic').checked && !gps) {
      return showMessage(msg, 'A synthetic GPS trace needs a GPS CSV file. Without one the session is stored with no GPS.');
    }

    const data = new FormData();
    data.append('video', video);
    if (gps) data.append('gps', gps);
    if (sidecar) data.append('sidecar', sidecar);
    data.append('role', $('#up-role').value);
    data.append('gps_source', $('#up-synthetic').checked ? 'synthetic' : 'recorded');
    data.append('include_in_map', $('#up-include').checked ? 'true' : 'false');

    showMessage(msg, null);
    const wrap = $('#up-progress-wrap');
    const text = $('#up-progress-text');
    wrap.hidden = false;
    const render = (f) => {
      clear($('#up-progress-slot')).append(progressBar(f, 'Upload progress'));
      text.textContent = f >= 1 ? 'Upload complete, waiting for the server...' : `Uploading ${Math.round(f * 100)}%`;
    };
    render(0);
    await withBusy(submit, async () => {
      try {
        await api.createSession(data, render);
        form.reset();
        wrap.hidden = true;
        emit('sessions-changed');
        showTab('sessions');
      } catch (e) {
        wrap.hidden = true;
        showMessage(msg, e.message);
      }
    });
  });
}
