// One function per API endpoint plus a single error-normalising request wrapper.

export class ApiError extends Error {
  constructor(message, { code = 'internal', status = 0, network = false } = {}) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.network = network;
  }
}

let connectionListener = () => {};
/** fn(true) after any reply from the server, fn(false) when it cannot be reached. */
export function onConnection(fn) { connectionListener = fn; }

function parseError(status, text) {
  try {
    const body = JSON.parse(text);
    if (body && body.error && typeof body.error.message === 'string') {
      return new ApiError(body.error.message, { code: body.error.code || 'internal', status });
    }
  } catch { /* not JSON */ }
  return new ApiError(`The server returned an unexpected response (HTTP ${status}).`, { status });
}

async function request(method, url, { json, form, signal } = {}) {
  const init = { method, signal, headers: { Accept: 'application/json' } };
  if (json !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(json);
  } else if (form) {
    init.body = form;
  }
  let res;
  try {
    res = await fetch(url, init);
  } catch (e) {
    if (e.name === 'AbortError') throw e;
    connectionListener(false);
    throw new ApiError('Cannot reach the server. Check that it is running and try again.',
      { code: 'network', network: true });
  }
  connectionListener(true);
  const text = await res.text();
  if (!res.ok) throw parseError(res.status, text);
  if (!text) return null;
  try { return JSON.parse(text); } catch {
    throw new ApiError('The server sent a reply that is not valid JSON.', { status: res.status });
  }
}

const bboxQuery = (bbox) => (bbox ? `?bbox=${bbox.map((v) => v.toFixed(6)).join(',')}` : '');
const sid = (id) => encodeURIComponent(id);

export const getHealth = () => request('GET', '/api/health');
export const getSystem = () => request('GET', '/api/system');
export const getStats = () => request('GET', '/api/stats');

export const listSessions = () => request('GET', '/api/sessions');
export const getSession = (id) => request('GET', `/api/sessions/${sid(id)}`);
export const getSessionStatus = (id) => request('GET', `/api/sessions/${sid(id)}/status`);
export const getDetections = (id) => request('GET', `/api/sessions/${sid(id)}/detections`);
export const getTrack = (id) => request('GET', `/api/sessions/${sid(id)}/track`);
export const reprocessSession = (id) => request('POST', `/api/sessions/${sid(id)}/reprocess`);
export const patchSession = (id, body) => request('PATCH', `/api/sessions/${sid(id)}`, { json: body });
export const deleteSession = (id) => request('DELETE', `/api/sessions/${sid(id)}`);

/** Multipart upload with progress; uses XMLHttpRequest because fetch has no upload progress. */
export function createSession(formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/sessions');
    xhr.setRequestHeader('Accept', 'application/json');
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    xhr.onerror = () => {
      connectionListener(false);
      reject(new ApiError('Cannot reach the server. The upload did not complete.',
        { code: 'network', network: true }));
    };
    xhr.onload = () => {
      connectionListener(true);
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText)); } catch {
          reject(new ApiError('The server sent a reply that is not valid JSON.', { status: xhr.status }));
        }
      } else {
        reject(parseError(xhr.status, xhr.responseText));
      }
    };
    xhr.send(formData);
  });
}

export const getPotholes = (bbox) => request('GET', `/api/potholes${bboxQuery(bbox)}`);
export const getPothole = (id) => request('GET', `/api/potholes/${sid(id)}`);
export const getCoverage = (bbox) => request('GET', `/api/coverage${bboxQuery(bbox)}`);
export const rebuildMap = (includeSynthetic) =>
  request('POST', '/api/map/rebuild', { json: { include_synthetic: includeSynthetic } });

export const getRoutesInfo = () => request('GET', '/api/routes/info');
export const getDemoRouteExample = () => request('GET', '/api/demo/route-example');
export const planRoutes = (origin, destination, avoidance) =>
  request('POST', '/api/routes', { json: { origin, destination, avoidance } });
