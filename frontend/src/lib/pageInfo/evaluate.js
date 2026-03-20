const evaluate = {
  title: 'Model Evaluation — Learning Guide',
  subtitle: 'Understanding how we measure whether a prediction model is good or bad.',
  sections: [
    {
      id: 'what-is-evaluation',
      heading: 'What is Model Evaluation?',
      icon: 'ClipboardCheck',
      blocks: [
        {
          type: 'text',
          content:
            'Model evaluation is like grading an exam. The model predicted certain bankline positions (its "answers"), and we compare those predictions against what actually happened (the "answer key"). The difference between prediction and reality tells us how good the model is.',
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Simple analogy',
          content:
            'Imagine you predicted tomorrow\'s temperature would be 25°C, and it turned out to be 27°C. Your error is 2°C. Evaluation metrics are just fancier versions of measuring this kind of error across many predictions.',
        },
      ],
    },
    {
      id: 'nse',
      heading: 'NSE — Nash-Sutcliffe Efficiency',
      icon: 'Gauge',
      blocks: [
        {
          type: 'text',
          content:
            'NSE answers the question: "Is this model better than just guessing the historical average every time?"',
        },
        {
          type: 'formula',
          content: 'NSE = 1 − Σ(observed − predicted)² / Σ(observed − mean_observed)²',
        },
        {
          type: 'interpretation',
          content:
            '• NSE = 1.0 → Perfect predictions (impossible in practice)\n• NSE > 0.7 → Good model\n• NSE > 0.5 → Acceptable\n• NSE = 0.0 → No better than guessing the average\n• NSE < 0.0 → Worse than the average — the model is harmful',
        },
        {
          type: 'metric-scale',
          config: {
            label: 'NSE',
            ranges: [
              { from: -1, to: 0, color: '#ef4444', label: 'Poor' },
              { from: 0, to: 0.5, color: '#eab308', label: 'Fair' },
              { from: 0.5, to: 0.7, color: '#f59e0b', label: 'OK' },
              { from: 0.7, to: 1.0, color: '#22c55e', label: 'Good' },
            ],
            unit: '',
          },
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Think of it this way',
          content:
            'If someone always guessed "the river bank will be at the average position," NSE = 0. Any positive NSE means the model learned something useful. The closer to 1, the better.',
        },
      ],
    },
    {
      id: 'kge',
      heading: 'KGE — Kling-Gupta Efficiency',
      icon: 'Target',
      blocks: [
        {
          type: 'text',
          content:
            'KGE grades the model on three separate "sub-tests" simultaneously, then combines them into one score:',
        },
        {
          type: 'formula',
          content: 'KGE = 1 − √[(r−1)² + (α−1)² + (β−1)²]\n\nwhere:\n  r = correlation (does it move in the right direction?)\n  α = variability ratio (does it bounce the right amount?)\n  β = bias ratio (is it systematically too high or too low?)',
        },
        {
          type: 'interpretation',
          content:
            '• KGE = 1.0 → Perfect on all three sub-tests\n• KGE > 0.7 → Good\n• KGE > 0.5 → Acceptable\n• KGE < 0 → Very poor',
        },
        {
          type: 'metric-scale',
          config: {
            label: 'KGE',
            ranges: [
              { from: -1, to: 0, color: '#ef4444', label: 'Poor' },
              { from: 0, to: 0.5, color: '#eab308', label: 'Fair' },
              { from: 0.5, to: 0.7, color: '#f59e0b', label: 'OK' },
              { from: 0.7, to: 1.0, color: '#22c55e', label: 'Good' },
            ],
            unit: '',
          },
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Why KGE over NSE?',
          content:
            'KGE decomposes performance into three components. A model can have good correlation but wrong variability. KGE catches these nuances that NSE misses. It is generally considered a more balanced metric.',
        },
      ],
    },
    {
      id: 'rmse',
      heading: 'RMSE — Root Mean Square Error',
      icon: 'AlertTriangle',
      blocks: [
        {
          type: 'text',
          content:
            'RMSE measures the average magnitude of prediction errors, with extra penalty for large errors.',
        },
        {
          type: 'formula',
          content: 'RMSE = √[ (1/n) × Σ(observed − predicted)² ]',
        },
        {
          type: 'interpretation',
          content:
            'RMSE is in the same unit as your data (metres). An RMSE of 150 means "on average, predictions are about 150 metres off, but large errors are penalized extra heavily."',
        },
        {
          type: 'metric-scale',
          config: {
            label: 'RMSE',
            ranges: [
              { from: 0, to: 200, color: '#22c55e', label: 'Good' },
              { from: 200, to: 500, color: '#eab308', label: 'Fair' },
              { from: 500, to: 1000, color: '#ef4444', label: 'Poor' },
            ],
            unit: 'm',
          },
        },
        {
          type: 'tip',
          variant: 'warning',
          title: 'Big errors hurt more',
          content:
            'Because errors are squared before averaging, one prediction that\'s 1000m off hurts the RMSE much more than ten predictions that are 100m off. Use RMSE when large errors are especially costly (e.g., infrastructure planning).',
        },
      ],
    },
    {
      id: 'mae',
      heading: 'MAE — Mean Absolute Error',
      icon: 'Ruler',
      blocks: [
        {
          type: 'text',
          content:
            'MAE is the simplest error metric — just the average of how far off each prediction was, ignoring direction.',
        },
        {
          type: 'formula',
          content: 'MAE = (1/n) × Σ|observed − predicted|',
        },
        {
          type: 'interpretation',
          content:
            'MAE is also in metres. An MAE of 120 means "on average, each prediction was about 120 metres away from reality." Unlike RMSE, all errors are weighted equally.',
        },
        {
          type: 'metric-scale',
          config: {
            label: 'MAE',
            ranges: [
              { from: 0, to: 150, color: '#22c55e', label: 'Good' },
              { from: 150, to: 400, color: '#eab308', label: 'Fair' },
              { from: 400, to: 1000, color: '#ef4444', label: 'Poor' },
            ],
            unit: 'm',
          },
        },
      ],
    },
    {
      id: 'rmse-vs-mae',
      heading: 'RMSE vs MAE — When to Use Which?',
      icon: 'Scale',
      blocks: [
        {
          type: 'text',
          content:
            'Both measure prediction error in metres, but they emphasize different things:',
        },
        {
          type: 'text',
          content:
            'RMSE penalizes large errors disproportionately. If most predictions are close but one is wildly off, RMSE will be much higher than MAE. Use RMSE when big mistakes are especially dangerous (e.g., placing a hospital based on bankline predictions).',
        },
        {
          type: 'text',
          content:
            'MAE treats all errors equally. Use MAE when you want a straightforward "average error" without extra emphasis on outliers.',
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Rule of thumb',
          content:
            'If RMSE is much larger than MAE for the same model, it means the model has some very large individual errors (outliers). If they\'re similar, errors are consistently sized.',
        },
      ],
    },
    {
      id: 'quantile-coverage',
      heading: 'Quantile Coverage',
      icon: 'BarChart3',
      blocks: [
        {
          type: 'text',
          content:
            'Quantile coverage measures how often actual values fall inside the predicted q10–q90 uncertainty band. The target is 80% (since the band spans from the 10th to 90th percentile).',
        },
        {
          type: 'formula',
          content: 'Coverage = (# of actuals inside [q10, q90]) / (total # of predictions) × 100%',
        },
        {
          type: 'interpretation',
          content:
            '• Coverage ≈ 80% → Well-calibrated uncertainty\n• Coverage > 80% → Band is too wide (model is too cautious / conservative)\n• Coverage < 80% → Band is too narrow (model is overconfident)',
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Only applies to TFT',
          content:
            'Baseline algorithms produce only point predictions (no q10/q90), so quantile coverage is only meaningful for the TFT model.',
        },
      ],
    },
    {
      id: 'color-thresholds',
      heading: 'Color Thresholds in the Table',
      icon: 'Palette',
      blocks: [
        {
          type: 'text',
          content:
            'Metric values in the table are color-coded to help you quickly spot good and bad performance:',
        },
        {
          type: 'color-legend',
          items: [
            { color: '#22c55e', label: 'Green — Good: NSE/KGE > 0.7, RMSE < 200m, MAE < 150m' },
            { color: '#eab308', label: 'Yellow — Acceptable: NSE/KGE 0.4–0.7, RMSE 200–500m, MAE 150–400m' },
            { color: '#ef4444', label: 'Red — Poor: NSE/KGE < 0.4, RMSE > 500m, MAE > 400m' },
          ],
        },
        {
          type: 'text',
          content:
            'These thresholds are guidelines for bankline distance prediction at this scale. A "poor" RMSE of 800m might be acceptable for a continent-wide model but is concerning for individual reaches of the Jamuna.',
        },
      ],
    },
    {
      id: 'val-test-split',
      heading: 'Train / Validation / Test Split',
      icon: 'Scissors',
      blocks: [
        {
          type: 'text',
          content:
            'The data is split into three non-overlapping time periods:',
        },
        {
          type: 'text',
          content:
            '• Train (≤ 2010) — The model learns patterns from this data\n• Validation (2011–2015) — Used to tune hyperparameters during training\n• Test (≥ 2016) — The final exam. The model never sees this during training',
        },
        {
          type: 'tip',
          variant: 'warning',
          title: 'Why separate sets?',
          content:
            'If you test a model on the same data it trained on, it will look artificially good (like giving a student the exact exam questions in advance). The test set is data the model has never seen — this gives an honest measure of real-world performance.',
        },
        {
          type: 'text',
          content:
            'On this page, you can view metrics for each split separately. Test metrics are the most important for judging real performance.',
        },
      ],
    },
    {
      id: 'blank-values',
      heading: 'Why Are Some Values Blank (—)?',
      icon: 'HelpCircle',
      blocks: [
        {
          type: 'text',
          content:
            'You may notice that some baseline algorithms show "—" (dashes) instead of metric values. This can happen for several reasons:',
        },
        {
          type: 'text',
          content:
            '1. Not trained yet — The baseline model hasn\'t been trained. Use the Training panel to train it.\n\n2. Algorithm failed — Some algorithms (especially ARIMA) may fail to converge on certain series. If too many series fail, overall metrics cannot be computed.\n\n3. Insufficient data — If a reach has too few observations, some algorithms cannot produce meaningful results.',
        },
        {
          type: 'tip',
          variant: 'warning',
          title: 'How to fix blank values',
          content:
            'Go to the Training panel and click "Train Baselines" to train all 10 baseline algorithms. Each will fit 100 models (50 reaches × 2 banks = 100 series). After training completes, refresh this page — values should appear.',
        },
        {
          type: 'text',
          content:
            'Note: Even after training, some algorithms may still show "—" for individual reaches where they genuinely could not produce a valid forecast. This is normal and expected.',
        },
      ],
    },
    {
      id: 'series-evaluated',
      heading: 'Number of Series Evaluated',
      icon: 'Hash',
      blocks: [
        {
          type: 'text',
          content:
            'The standard number is 100 series (50 reaches × 2 banks). If you see fewer, it means some series had insufficient historical data for evaluation.',
        },
        {
          type: 'text',
          content:
            'Each metric shown is the average across all evaluated series. Individual reaches may perform much better or worse than the average.',
        },
      ],
    },
    {
      id: 'interpreting-table',
      heading: 'Interpreting the Metrics Table',
      icon: 'Table',
      blocks: [
        {
          type: 'text',
          content:
            'The table shows all algorithms side by side with their metric values. Here\'s how to read it:',
        },
        {
          type: 'text',
          content:
            '• Higher is better for NSE and KGE (closer to 1.0)\n• Lower is better for RMSE and MAE (closer to 0)\n• Best values in each column are highlighted\n• The TFT column typically includes additional metrics like quantile coverage',
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'No single "best" metric',
          content:
            'A model might have the best NSE but not the best MAE. This is normal. Choose the metric that matches your use case: RMSE for safety-critical decisions, MAE for general planning, NSE/KGE for research.',
        },
        {
          type: 'text',
          content:
            'If all baselines show "—", you need to train them first. The TFT model is pre-trained and should always show values.',
        },
      ],
    },
  ],
};

export default evaluate;
