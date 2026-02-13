const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
  } catch {
    throw Object.assign(
      new Error('Cannot reach backend. Is it running? Start with: bash start.sh'),
      { status: 0 }
    );
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const msg =
      res.status === 503
        ? (err.detail || 'Model not loaded — run: python -m src.training.train')
        : (err.detail || 'Request failed');
    throw Object.assign(new Error(msg), { status: res.status });
  }
  return res.json();
}

export function getHealth() {
  return request('/health');
}

export function getPredictionForYear(year) {
  return request(`/predict/year/${year}`);
}

export function getSeriesHistory(reachId, bankSide, includeForecast = true) {
  return request(`/series/${reachId}/${bankSide}?include_forecast=${includeForecast}`);
}

export function getChangepoints(protectedOnly = false) {
  return request(`/anomaly/changepoints?protected_only=${protectedOnly}`);
}

export function getEvaluation(split = 'test') {
  return request(`/evaluate?split=${split}`);
}

export function postPredict(body) {
  return request('/predict', { method: 'POST', body: JSON.stringify(body) });
}

export function postPredictBaseline(body) {
  return request('/predict/baseline', { method: 'POST', body: JSON.stringify(body) });
}

export function postTrain(body = {}) {
  return request('/train', { method: 'POST', body: JSON.stringify(body) });
}

export function getTrainStatus(jobId) {
  return request(`/train/${jobId}/status`);
}

export function getTrainLogs(jobId, since = 0) {
  return request(`/train/${jobId}/logs?since=${since}`);
}
