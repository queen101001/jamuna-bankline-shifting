import { ALGORITHMS } from '@/lib/algorithms';

/**
 * Algorithm encyclopedia — detailed educational content for each of the 11 algorithms.
 * Keyed by algorithm key from algorithms.js. Merged with color/label at render time.
 */
export const ALGORITHM_DETAILS = {
  tft: {
    name: 'Temporal Fusion Transformer (TFT)',
    analogy: 'A weather station that reads multiple sensors at once, remembers which patterns mattered most, and gives you a range of possible outcomes instead of a single guess.',
    howItWorks: 'TFT is a deep learning architecture designed for multi-horizon time series forecasting. It uses attention mechanisms to learn which past time steps and input features matter most. It processes all 100 series (50 reaches x 2 banks) simultaneously, learning shared patterns across the entire river system.',
    strengths: ['Captures complex non-linear patterns', 'Produces probabilistic forecasts (q10, q50, q90)', 'Learns from all 100 series simultaneously'],
    weaknesses: ['Requires more training data and time', 'Can overfit on small datasets', 'Hard to interpret exactly why it made a prediction'],
    trustWhen: ['You have sufficient historical data (10+ years)', 'You need uncertainty estimates (quantile bands)', 'The river behavior is complex and non-linear'],
  },
  persistence: {
    name: 'Persistence (Naive Baseline)',
    analogy: 'Saying "tomorrow\'s weather will be the same as today" — it just repeats the last observed value forever.',
    howItWorks: 'Takes the most recent observed bankline distance and uses it as the prediction for all future time steps. No learning, no parameters — just the last known value repeated.',
    strengths: ['Simplest possible baseline', 'Impossible to overfit', 'Useful as a sanity check — any good model should beat this'],
    weaknesses: ['Cannot capture trends or patterns', 'Always wrong if the river is moving', 'No uncertainty estimates'],
    trustWhen: ['The bankline has been completely stable for many years', 'You need a quick reference baseline'],
  },
  linear: {
    name: 'Linear Extrapolation',
    analogy: 'Drawing a straight line through the data points and extending it forward — assumes the trend continues unchanged.',
    howItWorks: 'Fits a linear regression line (y = mx + b) through historical bankline distances over time. Predictions are made by extending this line into the future. Uses lag features for recursive multi-step forecasting.',
    strengths: ['Simple and interpretable', 'Works well when trends are genuinely linear', 'Very fast to train'],
    weaknesses: ['Cannot capture curves, cycles, or sudden changes', 'Assumes constant rate of change forever', 'Will diverge wildly over long horizons'],
    trustWhen: ['The bankline has been moving at a steady rate', 'You only need short-term predictions (1-5 years)', 'You want a simple interpretable baseline'],
  },
  arima: {
    name: 'ARIMA (AutoRegressive Integrated Moving Average)',
    analogy: 'A detective that looks at how past changes predict future changes — the classic statistical forecasting method used by economists and scientists for decades.',
    howItWorks: 'ARIMA models the time series as a combination of its own past values (Auto-Regressive), past forecast errors (Moving Average), and differencing to make the series stationary (Integrated). Parameters (p,d,q) control how much history to use.',
    strengths: ['Well-established statistical method with decades of research', 'Good for stationary or trend-stationary time series', 'Provides confidence intervals'],
    weaknesses: ['Assumes linear relationships', 'Struggles with complex seasonal patterns', 'May not converge for some series (shown as "—" in metrics)'],
    trustWhen: ['The bankline shows a clear, consistent trend', 'You need a statistically rigorous forecast', 'The series is relatively smooth without sudden jumps'],
  },
  random_forest: {
    name: 'Random Forest',
    analogy: 'Asking 100 different experts (decision trees) and averaging their opinions — the wisdom of the crowd.',
    howItWorks: 'Builds many decision trees, each trained on a random subset of the data and features. Each tree makes its own prediction, and the final forecast is the average of all trees. Uses lag features (past 3 values, rate of change, rolling mean) as inputs.',
    strengths: ['Captures non-linear patterns', 'Robust to outliers and noise', 'Hard to overfit'],
    weaknesses: ['Cannot extrapolate beyond the range of training data', 'Recursive multi-step forecasting can accumulate errors', 'No probabilistic output (point predictions only)'],
    trustWhen: ['The relationship between past and future is non-linear', 'You have enough training data', 'Short to medium-term forecasts'],
  },
  exp_smoothing: {
    name: 'Holt\'s Exponential Smoothing',
    analogy: 'Giving more weight to recent observations and less to old ones — like a fading memory that emphasizes the latest trends.',
    howItWorks: 'Holt\'s method extends simple exponential smoothing by adding a trend component. It has two smoothing parameters: one for the level (baseline value) and one for the trend (rate of change). Recent data points influence predictions more than old ones.',
    strengths: ['Naturally adapts to changing trends', 'Simple with only 2 parameters', 'Computationally efficient'],
    weaknesses: ['Assumes locally linear trends', 'Cannot capture complex non-linear patterns', 'May over-react to recent noise'],
    trustWhen: ['The bankline shows a clear recent trend', 'You want a lightweight, fast forecast', 'Recent behavior is more relevant than distant history'],
  },
  xgboost: {
    name: 'XGBoost (Extreme Gradient Boosting)',
    analogy: 'A team where each new member specializes in fixing the mistakes of previous members — iteratively getting better and better.',
    howItWorks: 'Builds decision trees sequentially. Each new tree focuses on the errors the previous trees made. Uses gradient descent to minimize prediction error. Features include lag values (lag_1, lag_2, lag_3), rate of change, and rolling mean.',
    strengths: ['Often top performer in tabular data competitions', 'Handles non-linear relationships well', 'Built-in regularization prevents overfitting'],
    weaknesses: ['Cannot extrapolate beyond training range', 'Requires careful hyperparameter tuning', 'Point predictions only — no uncertainty estimates'],
    trustWhen: ['You need the best possible point prediction accuracy', 'The data has complex non-linear patterns', 'Short to medium-term horizons'],
  },
  svr: {
    name: 'SVR (Support Vector Regression)',
    analogy: 'Finding the best possible tube around the data — predictions stay within this tube, allowing for some margin of error.',
    howItWorks: 'SVR finds a function that deviates from actual values by at most ε (epsilon) while being as flat as possible. It uses kernel functions to handle non-linear relationships. Points outside the ε-tube contribute to the error and shape the model.',
    strengths: ['Works well in high-dimensional spaces', 'Robust to outliers', 'Can model non-linear relationships with kernel trick'],
    weaknesses: ['Slower to train than tree-based methods', 'Sensitive to feature scaling', 'Harder to interpret than simpler models'],
    trustWhen: ['The dataset is moderate-sized (not too large)', 'You need robustness to outliers', 'The relationship is moderately non-linear'],
  },
  gradient_boosting: {
    name: 'Gradient Boosting',
    analogy: 'Similar to XGBoost — builds models sequentially, each one learning from the errors of the last, like a team improving together.',
    howItWorks: 'Gradient Boosting builds an ensemble of weak learners (typically decision trees) where each subsequent tree is trained to predict the residual errors of the combined ensemble so far. Uses gradient descent to minimize a loss function.',
    strengths: ['Flexible and powerful ensemble method', 'Can model complex non-linear relationships', 'Good default performance'],
    weaknesses: ['Slower than XGBoost (less optimized)', 'Can overfit with too many trees', 'Point predictions only'],
    trustWhen: ['You want a solid ML baseline', 'The data has non-linear patterns', 'Short to medium-term predictions'],
  },
  elastic_net: {
    name: 'Elastic Net',
    analogy: 'Linear regression with a built-in simplicity filter — it prevents the model from overcomplicating things by penalizing large weights.',
    howItWorks: 'Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization penalties on the linear regression coefficients. L1 encourages sparsity (some features get zero weight), while L2 shrinks all weights. The balance is controlled by a mixing parameter.',
    strengths: ['Handles correlated features well', 'Built-in feature selection via L1 penalty', 'Simple and interpretable'],
    weaknesses: ['Assumes linear relationships only', 'Cannot capture complex patterns', 'May underperform on non-linear data'],
    trustWhen: ['The bankline behavior is approximately linear', 'You want interpretable coefficients', 'Feature selection is important'],
  },
  knn: {
    name: 'KNN (K-Nearest Neighbours Regression)',
    analogy: 'Looking at the K most similar historical situations and averaging what happened next — learning from similar past experiences.',
    howItWorks: 'For each prediction, KNN finds the K training samples whose lag features are most similar to the current input, then averages their target values. No explicit model is built — it memorizes the training data and queries it at prediction time.',
    strengths: ['Simple and intuitive concept', 'No training phase (lazy learner)', 'Can capture local patterns'],
    weaknesses: ['Slow at prediction time for large datasets', 'Sensitive to the choice of K and distance metric', 'Cannot extrapolate beyond training data range'],
    trustWhen: ['Local patterns matter more than global trends', 'You want a non-parametric approach', 'The training dataset covers similar conditions'],
  },
};

/**
 * Build algorithm cards array for the block renderer.
 * Merges ALGORITHM_DETAILS with color/key from algorithms.js.
 */
export function getAlgorithmCards(keys) {
  const algos = keys
    ? ALGORITHMS.filter((a) => keys.includes(a.key))
    : ALGORITHMS;

  return algos.map((a) => {
    const details = ALGORITHM_DETAILS[a.key];
    return {
      key: a.key,
      color: a.color,
      name: details?.name || a.label,
      analogy: details?.analogy || '',
      howItWorks: details?.howItWorks || '',
      strengths: details?.strengths || [],
      weaknesses: details?.weaknesses || [],
      trustWhen: details?.trustWhen || [],
    };
  });
}

/** All 11 algorithm cards for the encyclopedia block */
export const ALL_ALGORITHM_CARDS = getAlgorithmCards();
