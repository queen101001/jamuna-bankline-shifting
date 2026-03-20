export const ALGORITHMS = [
  { key: 'tft', label: 'TFT', color: '#06b6d4' },
  { key: 'persistence', label: 'Persistence', color: '#94a3b8' },
  { key: 'linear', label: 'Linear', color: '#6366f1' },
  { key: 'arima', label: 'ARIMA', color: '#f97316' },
  { key: 'random_forest', label: 'Random Forest', color: '#22c55e' },
  { key: 'exp_smoothing', label: 'Holt', color: '#14b8a6' },
  { key: 'xgboost', label: 'XGBoost', color: '#a855f7' },
  { key: 'svr', label: 'SVR', color: '#ec4899' },
  { key: 'gradient_boosting', label: 'Gradient Boost', color: '#eab308' },
  { key: 'elastic_net', label: 'Elastic Net', color: '#f43f5e' },
  { key: 'knn', label: 'KNN', color: '#8b5cf6' },
];

export const ALGO_COLOR_MAP = Object.fromEntries(
  ALGORITHMS.map((a) => [a.key, a.color])
);
