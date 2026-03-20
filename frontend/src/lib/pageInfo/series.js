const series = {
  title: 'Series Detail — Learning Guide',
  subtitle: 'How to read and interpret the time series chart and forecast table.',
  sections: [
    {
      id: 'chart-reading',
      heading: 'How to Read the Chart',
      icon: 'LineChart',
      blocks: [
        {
          type: 'text',
          content:
            'The chart shows the complete history and future prediction for one reach and one bank side. Here is an annotated guide to every visual element:',
        },
        { type: 'diagram', component: 'chartReading' },
        {
          type: 'text',
          content:
            '• Solid dots with line — Real observed historical data\n• Dashed line — Model\'s forecast (predicted future)\n• Shaded band — Uncertainty range (where reality will probably fall)\n• Vertical reference line — Boundary between history and forecast\n• Orange diamond — Changepoint (a detected structural break)',
        },
      ],
    },
    {
      id: 'axes',
      heading: 'X-Axis and Y-Axis',
      icon: 'Move',
      blocks: [
        {
          type: 'text',
          content:
            'X-Axis (horizontal) = Year. Spans from 1991 (first observation) through up to 2100 (furthest forecast). Not every year has a data point — satellite measurements were taken at irregular intervals.',
        },
        {
          type: 'text',
          content:
            'Y-Axis (vertical) = Bankline distance in metres. This is measured from a fixed reference point on the riverbank to the water\'s edge. A larger number means the bank is further from the reference. Whether that means erosion or deposition depends on the bank side (see Sign Convention).',
        },
      ],
    },
    {
      id: 'historical-vs-forecast',
      heading: 'Historical vs Forecast',
      icon: 'Split',
      blocks: [
        {
          type: 'text',
          content:
            'The chart is divided into two zones by a vertical reference line at the year 2020:',
        },
        {
          type: 'text',
          content:
            'Left of the line — Historical data (1991–2020): These are real satellite-derived measurements. Solid dots connected by a solid line. This is ground truth.\n\nRight of the line — Forecast (2021+): These are the model\'s predictions. Shown as a dashed line with smaller dots. These are estimates, not measurements.',
        },
        {
          type: 'tip',
          variant: 'warning',
          title: 'Predictions are not facts',
          content:
            'Everything to the right of the reference line is a model\'s best guess. The further into the future, the less reliable the prediction becomes. Always check the uncertainty band width.',
        },
      ],
    },
    {
      id: 'quantile-bands',
      heading: 'Quantile Bands (Uncertainty)',
      icon: 'BarChart3',
      blocks: [
        {
          type: 'text',
          content:
            'The TFT model doesn\'t just predict a single number — it predicts a range of possible outcomes using three quantile lines:',
        },
        { type: 'diagram', component: 'quantile' },
        {
          type: 'text',
          content:
            '• q50 (middle, bold) — The median prediction. "My best guess is this value."\n• q10 (lower) — The optimistic bound. "There\'s only a 10% chance it will be below this."\n• q90 (upper) — The pessimistic bound. "There\'s only a 10% chance it will be above this."',
        },
        {
          type: 'text',
          content:
            'The shaded area between q10 and q90 is the 80% prediction interval — roughly 80% of actual future values should fall within this band.',
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Why 80% and not 100%?',
          content:
            'A 100% prediction interval would be infinitely wide and useless. The 80% interval gives a useful range while acknowledging that 1 in 5 observations may fall outside it.',
        },
      ],
    },
    {
      id: 'bands-widen',
      heading: 'Why Bands Widen Over Time',
      icon: 'TrendingUp',
      blocks: [
        {
          type: 'text',
          content:
            'You may notice that the shaded uncertainty band gets wider the further into the future you look. This is expected and healthy — it means the model is being honest about growing uncertainty.',
        },
        {
          type: 'text',
          content:
            'Reasons bands widen:\n• Rolling forecasts compound small errors at each step\n• The river is inherently unpredictable over long horizons\n• Climate change and human intervention can alter patterns in unexpected ways',
        },
        {
          type: 'tip',
          variant: 'warning',
          title: 'Be cautious with 2050+ forecasts',
          content:
            'Predictions beyond 2050 should be treated as rough directional estimates, not precise values. The band width tells you how uncertain the model is.',
        },
      ],
    },
    {
      id: 'changepoint-markers',
      heading: 'Changepoint Markers',
      icon: 'Diamond',
      blocks: [
        {
          type: 'text',
          content:
            'Orange diamond shapes on the chart mark changepoints — years where the PELT algorithm detected a sudden structural change in the bankline\'s behavior.',
        },
        {
          type: 'text',
          content:
            'A changepoint means the river\'s "personality" changed at that year. Before the changepoint, the bank might have been bouncing wildly; after it, the bank might have become calm (or vice versa).',
        },
        {
          type: 'text',
          content:
            'Common causes of changepoints:\n• Bank protection structures installed (revetments, groynes)\n• Major flood events that reshaped the channel\n• Upstream dam construction or flow changes\n• Natural avulsion (river shifts course)',
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
            'When the variance (bounciness) of bankline distances drops by 70% or more after a changepoint, the system flags it as a protection signature. This strongly suggests that man-made bank protection was installed.',
        },
        {
          type: 'text',
          content:
            'Types of bank protection on the Jamuna:\n• Revetments — armoured embankments along the bank\n• Groynes — structures jutting into the river to deflect flow\n• Spurs — similar to groynes, built at angles\n• Concrete blocks — placed along vulnerable sections',
        },
        {
          type: 'tip',
          variant: 'erosion',
          title: 'Impact on predictions',
          content:
            'Protected banks behave very differently from natural banks. Models trained on historical erosion patterns may overpredict movement for protected reaches. Look for the orange "Protected" badge.',
        },
      ],
    },
    {
      id: 'forecast-table',
      heading: 'Forecast Table',
      icon: 'Table',
      blocks: [
        {
          type: 'text',
          content:
            'Below the chart, a table shows the numerical predictions for 5 future time steps.',
        },
        {
          type: 'text',
          content:
            'For TFT: Three columns per step — q10, q50, q90. These correspond to the three quantile lines on the chart.\n\nFor baseline algorithms: Only one column per step (point prediction). Baselines do not produce uncertainty estimates — they give only a single best guess.',
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Why do baselines show only one value?',
          content:
            'Most traditional algorithms (Linear, ARIMA, Random Forest, etc.) are designed to output a single predicted value, not a probability distribution. Only the TFT model was specifically designed to produce quantile forecasts.',
        },
      ],
    },
    {
      id: 'baseline-vs-tft',
      heading: 'Baseline vs TFT',
      icon: 'Scale',
      blocks: [
        {
          type: 'text',
          content:
            'The fundamental difference between TFT and baseline algorithms:',
        },
        {
          type: 'text',
          content:
            'TFT (Temporal Fusion Transformer):\n• Probabilistic — gives a range of outcomes (q10, q50, q90)\n• Learns from all 100 series simultaneously\n• Uses attention to focus on the most relevant past time steps\n• More complex, but captures subtler patterns',
        },
        {
          type: 'text',
          content:
            'Baselines (10 algorithms):\n• Point predictions only — a single number per time step\n• Each series is modeled independently\n• Simpler and faster, but may miss complex patterns\n• Serve as benchmarks to validate TFT\'s added value',
        },
        {
          type: 'tip',
          variant: 'success',
          title: 'When does TFT shine?',
          content:
            'TFT tends to outperform baselines when the river behavior is complex, non-linear, and when having uncertainty estimates (the prediction band) is important for decision-making.',
        },
      ],
    },
  ],
};

export default series;
