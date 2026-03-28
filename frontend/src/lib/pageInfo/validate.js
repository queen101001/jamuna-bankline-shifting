const validate = {
  title: 'Validation — Learning Guide',
  subtitle: 'How to test model predictions against your own real-world data.',
  sections: [
    {
      id: 'what-is-validation',
      heading: 'What is Validation?',
      icon: 'FileCheck',
      blocks: [
        {
          type: 'text',
          content:
            'Validation lets you test the models against YOUR own data. The models were trained on historical measurements from 1991–2020. If you have newer measurements (2021–2025), you can upload them here to see how well the predictions matched reality.',
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Why is this important?',
          content:
            'Built-in evaluation metrics (on the Evaluate page) test against held-out training data. Validation with YOUR data tests against truly independent measurements that the model has never been influenced by. This is the ultimate test of a model.',
        },
      ],
    },
    {
      id: 'excel-format',
      heading: 'Required Excel Format',
      icon: 'FileSpreadsheet',
      blocks: [
        {
          type: 'text',
          content:
            'Your Excel file (.xlsx) must match a specific 2-row header format. The first row contains reach IDs, and the second row contains bank side labels.',
        },
        { type: 'diagram', component: 'excelFormat' },
        {
          type: 'text',
          content:
            '• Row 1: "Year" in column A, then reach IDs (R01, R01, R02, R02, …) — each reach appears twice (left and right bank)\n• Row 2: Empty in column A, then alternating "Left" and "Right"\n• Data rows: Year in column A (e.g., 2021), then distance values in metres\n• Distance values should be positive numbers (the system handles sign convention automatically)',
        },
        {
          type: 'tip',
          variant: 'warning',
          title: 'Common mistakes',
          content:
            'Make sure reach IDs match exactly (R01, not r01 or Reach1). Bank labels must be "Left" and "Right" (case-sensitive). Years should be numbers (2021), not text.',
        },
      ],
    },
    {
      id: 'sign-convention',
      heading: 'Sign Convention',
      icon: 'ArrowLeftRight',
      blocks: [
        {
          type: 'text',
          content:
            'The sign convention is the same for both banks: negative values indicate erosion (bank moving toward the river) and positive values indicate deposition (bank growing outward). No sign transformation is applied to uploaded data.',
        },
        {
          type: 'text',
          content:
            'Provide your measured distances with the correct sign in the Excel file. Negative = erosion, positive = deposition for both left and right banks.',
        },
      ],
    },
    {
      id: 'algorithm-tabs',
      heading: 'Algorithm Tabs',
      icon: 'Layers',
      blocks: [
        {
          type: 'text',
          content:
            'After uploading data, you can switch between algorithm tabs to see how each model performed against your measurements:',
        },
        {
          type: 'text',
          content:
            '• Individual algorithm tabs — Show that algorithm\'s predictions vs your data\n• "All" tab — Overlays all 11 algorithms on one chart for direct comparison\n• Each tab shows error metrics specific to that algorithm',
        },
      ],
    },
    {
      id: 'error-metrics',
      heading: 'Error Metrics',
      icon: 'BarChart3',
      blocks: [
        {
          type: 'text',
          content:
            'For each algorithm, the validation page computes:',
        },
        {
          type: 'text',
          content:
            '• MAE (Mean Absolute Error) — Average distance between prediction and your data (metres)\n• RMSE (Root Mean Square Error) — Like MAE but penalizes large errors more heavily\n• Max Error — The single worst prediction across all reaches and years',
        },
        {
          type: 'text',
          content:
            'Practical interpretation:\n• < 100m — Excellent. Predictions closely match reality\n• 100–500m — Acceptable. Reasonable for river bankline prediction\n• > 500m — The model may not be suitable for this reach',
        },
      ],
    },
    {
      id: 'best-model',
      heading: 'Best Model Detection',
      icon: 'Trophy',
      blocks: [
        {
          type: 'text',
          content:
            'The system automatically identifies which algorithm had the lowest MAE against your uploaded data and marks it with a green "Best" badge.',
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Context matters',
          content:
            'The "best" model depends entirely on what data you uploaded. If you uploaded data for only 5 reaches, the best model is determined by performance on just those 5 reaches. Upload more data for a more comprehensive comparison.',
        },
        {
          type: 'text',
          content:
            'The best model for your specific data may differ from the best model on the Evaluate page (which uses the built-in test set). This is normal and valuable information.',
        },
      ],
    },
    {
      id: 'reading-charts',
      heading: 'Reading the Validation Charts',
      icon: 'LineChart',
      blocks: [
        {
          type: 'text',
          content:
            'Each chart shows a comparison between your uploaded data and the model\'s predictions:',
        },
        {
          type: 'text',
          content:
            '• Cyan solid line — Your observed data (ground truth)\n• Dashed colored line — The model\'s prediction\n• Gap between lines — The prediction error at each year\n• Closer lines = better predictions',
        },
        {
          type: 'text',
          content:
            'You can hover over data points to see exact values. The visual gap between lines gives an intuitive sense of how far off the predictions were.',
        },
      ],
    },
    {
      id: 'what-to-do',
      heading: 'What to Do with Results',
      icon: 'Lightbulb',
      blocks: [
        {
          type: 'text',
          content:
            'After validation, use the results to guide your decisions:',
        },
        {
          type: 'text',
          content:
            '• All models accurate (< 100m) → High confidence in forecasts for this reach\n• TFT best, baselines worse → TFT captures patterns that simpler models miss\n• A baseline best → The pattern at this reach may be simple (linear, persistent)\n• All models poor (> 500m) → The bankline behavior changed in ways not captured by training data',
        },
        {
          type: 'tip',
          variant: 'warning',
          title: 'When all models fail',
          content:
            'If all algorithms perform poorly, it may indicate a regime change — the river is behaving fundamentally differently than it did during 1991–2020. This could be due to new bank protection, upstream dam changes, or climate-driven shifts. Consider updating the training data with newer observations.',
        },
      ],
    },
  ],
};

export default validate;
