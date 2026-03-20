const anomaly = {
  title: 'Anomaly Detection — Learning Guide',
  subtitle: 'How we detect sudden changes in river behavior and identify bank protection.',
  sections: [
    {
      id: 'what-is-anomaly',
      heading: 'What is Anomaly Detection?',
      icon: 'AlertTriangle',
      blocks: [
        {
          type: 'text',
          content:
            'Anomaly detection is like a smoke detector for the river. Instead of looking for smoke, it scans the timeline of bankline measurements and alerts you when the pattern suddenly changed.',
        },
        {
          type: 'text',
          content:
            'The key question it answers: "Was there a specific year when this bank started behaving very differently?" If yes, we want to know when, and why.',
        },
      ],
    },
    {
      id: 'pelt-algorithm',
      heading: 'The PELT Algorithm',
      icon: 'Zap',
      blocks: [
        {
          type: 'text',
          content:
            'PELT stands for Pruned Exact Linear Time. It is a mathematical algorithm that scans through a time series and finds the exact points where the statistical properties (like variance) changed abruptly.',
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Simple analogy',
          content:
            'Imagine driving on a smooth highway and suddenly hitting a gravel road. You\'d immediately notice the change. PELT does the same thing mathematically — it finds the exact point where the "road surface" (bankline behavior) changed.',
        },
        {
          type: 'text',
          content:
            'PELT is efficient (fast even on long time series) and exact (it finds the optimal changepoints, not approximations).',
        },
      ],
    },
    {
      id: 'variance',
      heading: 'What is Variance?',
      icon: 'Activity',
      blocks: [
        {
          type: 'text',
          content:
            'Variance measures how much the bankline bounces around from year to year. Think of it as the "wobbliness" of the bank.',
        },
        {
          type: 'text',
          content:
            '• High variance → The bank moves a lot each year (unpredictable, unstable)\n• Low variance → The bank stays roughly in the same place (stable, predictable)',
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Road analogy',
          content:
            'High variance is like a bumpy dirt road — lots of ups and downs. Low variance is like a smooth highway — consistent and stable.',
        },
      ],
    },
    {
      id: 'variance-reduction',
      heading: 'Variance Reduction',
      icon: 'TrendingDown',
      blocks: [
        {
          type: 'text',
          content:
            'When a changepoint is detected, we compare the variance before and after it:',
        },
        {
          type: 'formula',
          content: 'Variance Reduction = (variance_before − variance_after) / variance_before × 100%',
        },
        {
          type: 'text',
          content:
            '• 90% reduction → The bank went from very bouncy to almost completely calm\n• 50% reduction → Moderate stabilization\n• 0% or negative → The bank became MORE unstable (variance increased)',
        },
        {
          type: 'text',
          content:
            'The variance before and after are shown as separate columns in the table, measured in square metres (m²).',
        },
      ],
    },
    {
      id: 'protection-signatures',
      heading: 'Protection Signatures',
      icon: 'Shield',
      blocks: [
        {
          type: 'text',
          content:
            'A protection signature is flagged when variance reduction is ≥ 70% after a changepoint. This almost always indicates that man-made bank protection was installed at that location.',
        },
        {
          type: 'text',
          content:
            'Types of bank protection used on the Jamuna River:\n• Revetments — Armoured stone or concrete walls along the bank\n• Groynes — Structures extending into the river to redirect flow\n• Spurs — Angular structures deflecting erosive currents\n• Geobags — Large sand-filled bags placed along vulnerable banks',
        },
        {
          type: 'tip',
          variant: 'success',
          title: 'Real-world meaning',
          content:
            'Protection signatures are practically useful. If a government agency invested in bank protection at a certain location, we can verify whether it actually worked by checking if the variance dropped dramatically after installation.',
        },
      ],
    },
    {
      id: 'why-70-percent',
      heading: 'Why 70% Threshold?',
      icon: 'HelpCircle',
      blocks: [
        {
          type: 'text',
          content:
            'The 70% threshold was determined empirically from the Jamuna dataset. Natural changes in river behavior — floods, seasonal patterns, gradual channel migration — rarely cause variance reductions above 70%.',
        },
        {
          type: 'text',
          content:
            'When we see a ≥70% drop in variance, it is almost always associated with documented bank protection construction. This makes it a reliable indicator.',
        },
        {
          type: 'tip',
          variant: 'warning',
          title: 'Not 100% certain',
          content:
            'While ≥70% strongly suggests protection, rare natural events (e.g., a massive sandbar forming) could also stabilize a bank. Cross-reference with local engineering records when possible.',
        },
      ],
    },
    {
      id: 'reading-table',
      heading: 'Reading the Table',
      icon: 'Table',
      blocks: [
        {
          type: 'text',
          content: 'Each row in the anomaly table represents one detected changepoint:',
        },
        {
          type: 'text',
          content:
            '• Reach — Which cross-section (R01–R50)\n• Bank — Left or Right\n• Changepoint Year — The year the behavior changed\n• Variance Before — How bouncy the bank was before (m²)\n• Variance After — How bouncy it became after (m²)\n• Variance Reduction — Percentage drop in bounciness\n• Protection — Whether it meets the ≥70% threshold',
        },
        {
          type: 'text',
          content:
            'Not every reach has a changepoint. Some banks have been consistently erosive or stable throughout the measurement period.',
        },
      ],
    },
    {
      id: 'sorting-filtering',
      heading: 'Sorting & Filtering',
      icon: 'Filter',
      blocks: [
        {
          type: 'text',
          content:
            'Click any column header to sort the table by that column. Click again to reverse the sort order.',
        },
        {
          type: 'text',
          content:
            'The "Protected only" filter shows only rows where protection signatures were detected. This is useful for quickly identifying all bank-protected locations along the river.',
        },
      ],
    },
  ],
};

export default anomaly;
