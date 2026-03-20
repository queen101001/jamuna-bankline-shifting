import { ALL_ALGORITHM_CARDS } from './shared';

const compare = {
  title: 'Model Comparison — Learning Guide',
  subtitle: 'How to compare predictions from different algorithms and decide which to trust.',
  sections: [
    {
      id: 'why-compare',
      heading: 'Why Compare Models?',
      icon: 'GitCompare',
      blocks: [
        {
          type: 'text',
          content:
            'No single algorithm is universally the best. Different reaches of the river may have different characteristics — some erode linearly, some have complex non-linear patterns, some are stabilized by protection works.',
        },
        {
          type: 'text',
          content:
            'By comparing multiple algorithms on the same reach, you can:\n• See which model fits best for that specific location\n• Identify where models agree (higher confidence) and disagree (higher uncertainty)\n• Make more informed decisions by considering multiple perspectives',
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Wisdom of the crowd',
          content:
            'When all 11 algorithms predict similar values, you can be more confident in the forecast. When they diverge wildly, the true future is genuinely uncertain.',
        },
      ],
    },
    {
      id: 'reading-chart',
      heading: 'Reading the Comparison Chart',
      icon: 'LineChart',
      blocks: [
        {
          type: 'text',
          content:
            'The chart overlays forecast lines from all selected algorithms on the same axes:',
        },
        {
          type: 'text',
          content:
            '• Solid thick line — TFT (the primary deep learning model)\n• Dashed/dotted lines — Baseline algorithms (each in a different color)\n• Historical data — Shown as solid dots/line for reference\n• Each line extends from 2021 (or earlier for historical) into the future',
        },
        {
          type: 'text',
          content:
            'The colors match the algorithm legend shown below the chart. Each algorithm has a unique color so you can track which prediction belongs to which model.',
        },
      ],
    },
    {
      id: 'toggle-buttons',
      heading: 'Toggle Buttons',
      icon: 'ToggleLeft',
      blocks: [
        {
          type: 'text',
          content:
            'Below the chart, toggle buttons let you show or hide individual algorithms. Each button shows the algorithm\'s color dot and name.',
        },
        {
          type: 'text',
          content:
            '• Click to toggle an algorithm on/off\n• Active buttons have a colored border matching the algorithm\n• Start with all visible, then hide to focus on specific comparisons\n• "TFT" is always available; baselines may show "not trained" if not yet fitted',
        },
      ],
    },
    {
      id: 'diverging-predictions',
      heading: 'When Predictions Diverge',
      icon: 'GitFork',
      blocks: [
        {
          type: 'text',
          content:
            'Pay attention to how spread out the prediction lines are:',
        },
        {
          type: 'text',
          content:
            '• Lines close together → Models agree → Higher confidence in the forecast\n• Lines spread apart → Models disagree → Genuine uncertainty about the future\n• One outlier line → That algorithm may not suit this reach',
        },
        {
          type: 'tip',
          variant: 'warning',
          title: 'Divergence grows over time',
          content:
            'Predictions tend to agree more for near-term forecasts (2021–2025) and diverge more for long-term ones (2050+). This is normal — uncertainty naturally grows with the forecast horizon.',
        },
      ],
    },
    {
      id: 'reach-selector',
      heading: 'Reach & Bank Selectors',
      icon: 'MapPin',
      blocks: [
        {
          type: 'text',
          content:
            'Use the dropdown selectors to choose which reach (R01–R50) and bank side (Left/Right) to compare. Each combination shows a completely different set of predictions because each location has unique erosion patterns.',
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Try different reaches',
          content:
            'You may find that the TFT dominates at one reach but a simple baseline like Linear wins at another. This is expected — river dynamics vary along its length.',
        },
      ],
    },
    {
      id: 'metrics-table',
      heading: 'Metrics Table',
      icon: 'Table',
      blocks: [
        {
          type: 'text',
          content:
            'Below the chart, a metrics table shows evaluation scores for each algorithm at the selected reach:',
        },
        {
          type: 'text',
          content:
            '• NSE, KGE — Higher is better (closer to 1.0 is best)\n• RMSE, MAE — Lower is better (closer to 0 is best)\n• Best value in each column is highlighted',
        },
        {
          type: 'text',
          content:
            'These metrics are computed on the test set (2016–2020) for the specific reach you\'re viewing, not the global average. A model that performs well overall might struggle on a particular reach.',
        },
      ],
    },
    {
      id: 'deciding-trust',
      heading: 'Deciding Which Model to Trust',
      icon: 'ShieldCheck',
      blocks: [
        {
          type: 'text',
          content:
            'There\'s no single answer, but here are guidelines:',
        },
        {
          type: 'text',
          content:
            '• For infrastructure planning (bridges, embankments) → Use the model with lowest RMSE. Big errors are costly.\n• For general forecasting → Use the model with lowest MAE. Average accuracy matters most.\n• For scientific research → Use the model with highest NSE or KGE. Statistical quality matters.\n• When you need uncertainty ranges → Only TFT provides quantile bands.',
        },
        {
          type: 'tip',
          variant: 'success',
          title: 'Practical rule',
          content:
            'If TFT has the best metrics AND provides uncertainty bands, use TFT. If a simpler baseline performs equally well, prefer the simpler model — it\'s easier to explain and less likely to break in unexpected ways.',
        },
      ],
    },
    {
      id: 'algorithm-encyclopedia',
      heading: 'Algorithm Encyclopedia',
      icon: 'BookOpen',
      blocks: [
        {
          type: 'text',
          content:
            'Detailed information about all 11 algorithms used in this application. Click any card to expand and learn about how it works, its strengths, weaknesses, and when to trust it.',
        },
        {
          type: 'algorithms',
          algorithms: ALL_ALGORITHM_CARDS,
          expanded: true,
        },
      ],
    },
  ],
};

export default compare;
