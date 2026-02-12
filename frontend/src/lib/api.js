const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw Object.assign(new Error(err.detail || 'Request failed'), { status: res.status });
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
