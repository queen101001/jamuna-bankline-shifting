import { ALL_ALGORITHM_CARDS } from './shared';

const dashboard = {
  title: 'Dashboard — Learning Guide',
  subtitle: 'Everything you need to understand this page, from scratch.',
  sections: [
    {
      id: 'jamuna-river',
      heading: 'What is the Jamuna River?',
      icon: 'Waves',
      blocks: [
        {
          type: 'text',
          content:
            'The Jamuna is one of the world\'s most dynamic braided rivers, flowing through Bangladesh. It is the main distributary of the Brahmaputra and carries enormous volumes of water and sediment. Its banks shift dramatically — sometimes hundreds of metres in a single year — destroying farmland, homes, and infrastructure.',
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Why does this matter?',
          content:
            'Millions of people live along the Jamuna. Predicting where the bankline will be in 5, 10, or 50 years helps governments plan flood defenses, relocate communities, and build infrastructure in safe locations.',
        },
        {
          type: 'text',
          content:
            'This application uses machine learning to forecast future bankline positions based on 27 years of satellite-derived measurements (1991–2020).',
        },
      ],
    },
    {
      id: 'reaches',
      heading: 'What are Reaches?',
      icon: 'MapPin',
      blocks: [
        {
          type: 'text',
          content:
            'The river is divided into 50 cross-sections called "reaches" (R01 to R50), spaced along its length from north to south. Each reach is measured on two sides:',
        },
        {
          type: 'text',
          content:
            '• Left Bank — the bank on the left when facing downstream\n• Right Bank — the bank on the right when facing downstream',
        },
        {
          type: 'text',
          content:
            'This gives us 50 × 2 = 100 individual time series to predict. Each series tracks how far the bankline is from a fixed reference point over time.',
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Think of it like this',
          content:
            'Imagine 50 rulers laid across the river at regular intervals. Each ruler measures the distance from a fixed post to the water\'s edge on both sides. We track how these measurements change year by year.',
        },
      ],
    },
    {
      id: 'sign-convention',
      heading: 'Sign Convention (+ and −)',
      icon: 'ArrowLeftRight',
      blocks: [
        {
          type: 'text',
          content:
            'The sign (positive or negative) of a distance value tells you whether the bank is eroding or growing. This works differently for left and right banks:',
        },
        { type: 'diagram', component: 'erosion' },
        {
          type: 'text',
          content:
            'Left Bank:\n  • Positive (+) distance → Erosion (bank moving toward river, land is lost)\n  • Negative (−) distance → Deposition (bank growing outward, land is gained)\n\nRight Bank:\n  • Negative (−) distance → Erosion (bank moving toward river, land is lost)\n  • Positive (+) distance → Deposition (bank growing outward, land is gained)',
        },
        {
          type: 'tip',
          variant: 'warning',
          title: 'Why is it reversed?',
          content:
            'Because the left and right banks face opposite directions. Erosion always means "the bank moved toward the river center." On the left bank that\'s the positive direction; on the right bank it\'s the negative direction.',
        },
      ],
    },
    {
      id: 'color-coding',
      heading: 'Color Coding',
      icon: 'Palette',
      blocks: [
        {
          type: 'text',
          content: 'This application uses two color systems:',
        },
        {
          type: 'heading',
          content: '1. Erosion / Deposition Colors',
        },
        {
          type: 'color-legend',
          items: [
            { color: 'var(--erosion)', label: 'Red — Erosion (land loss, danger, bank retreating)' },
            { color: 'var(--deposition)', label: 'Green — Deposition (land gain, bank growing outward)' },
          ],
        },
        {
          type: 'heading',
          content: '2. Metric Quality Colors (on Evaluate page)',
        },
        {
          type: 'color-legend',
          items: [
            { color: '#22c55e', label: 'Green — Good performance' },
            { color: '#eab308', label: 'Yellow — Acceptable performance' },
            { color: '#ef4444', label: 'Red — Poor performance' },
          ],
        },
      ],
    },
    {
      id: 'reach-grid',
      heading: 'Reading the Reach Grid',
      icon: 'Grid3X3',
      blocks: [
        {
          type: 'text',
          content:
            'The main area of the dashboard shows a 5×10 grid of cards — one for each of the 50 reaches. Each card contains:',
        },
        {
          type: 'text',
          content:
            '• Reach label (e.g., R01, R02 …)\n• Two horizontal bars — one for the left bank, one for the right bank\n• Bar length represents the predicted bankline distance for the selected year\n• Bar color: red for erosion, green for deposition\n• Hover over a bar to see detailed values (q10, q50, q90 for TFT)',
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Click any card',
          content:
            'Clicking a reach card navigates to the Series Detail page where you can see the full historical chart and forecast for that reach and bank.',
        },
      ],
    },
    {
      id: 'year-selector',
      heading: 'Year Selector',
      icon: 'Calendar',
      blocks: [
        {
          type: 'text',
          content:
            'The year selector at the top lets you choose any year from 2021 to 2100. The predictions shown in the grid update accordingly.',
        },
        {
          type: 'text',
          content:
            '• Years 2021–2025 show a "Direct" badge — these are direct forecasts from the model in a single pass.\n• Years 2026–2100 show a "Rolling" badge — these are rolling forecasts where predictions feed back into the model iteratively.',
        },
        {
          type: 'tip',
          variant: 'warning',
          title: 'Rolling forecasts are less certain',
          content:
            'Each step in a rolling forecast uses the previous prediction as input, so small errors can compound over time. Predictions for 2090 are inherently less reliable than predictions for 2022.',
        },
      ],
    },
    {
      id: 'forecast-modes',
      heading: 'Direct vs Rolling Forecasts',
      icon: 'GitBranch',
      blocks: [
        {
          type: 'text',
          content:
            'The model uses two different strategies depending on how far into the future you\'re predicting:',
        },
        { type: 'diagram', component: 'forecastModes' },
        {
          type: 'text',
          content:
            'Direct Forecast (2021–2025): The model takes real historical data as input and predicts the next 5 years in a single pass. These predictions are the most reliable because they\'re based entirely on real observations.',
        },
        {
          type: 'text',
          content:
            'Rolling Forecast (2026–2100): To predict beyond 2025, the model takes its own predictions and feeds them back as if they were real data, then predicts 5 more years. This process repeats until the target year. Each cycle adds a little more uncertainty.',
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
            'The tabs at the top let you switch between different prediction algorithms:',
        },
        {
          type: 'text',
          content:
            '• TFT (Temporal Fusion Transformer) — the primary deep learning model. This is the most sophisticated algorithm, trained on all 100 series simultaneously. It provides probabilistic forecasts (a range of outcomes, not just a single number).\n\n• Baseline algorithms (10 others) — simpler models for comparison. They help verify whether the TFT is actually better, and by how much.',
        },
        {
          type: 'tip',
          variant: 'info',
          title: 'Why have baselines?',
          content:
            'In science, you always compare against simpler methods. If a complex deep learning model can\'t beat a simple straight line, then the complexity isn\'t worth it. Baselines keep the TFT honest.',
        },
      ],
    },
    {
      id: 'protected-badge',
      heading: '"Protected" Badge',
      icon: 'ShieldAlert',
      blocks: [
        {
          type: 'text',
          content:
            'Some reach cards show an orange "Protected" badge. This means the PELT anomaly detection algorithm found a protection signature at this reach.',
        },
        {
          type: 'text',
          content:
            'A protection signature means the bankline\'s variability (how much it bounces around year to year) suddenly dropped by ≥70% at some point in time. This usually indicates that bank protection structures — such as revetments, groynes, or concrete spurs — were installed at that location.',
        },
        {
          type: 'tip',
          variant: 'warning',
          title: 'Why does this matter for predictions?',
          content:
            'If a bank is protected, the model\'s predictions based on historical erosion patterns may not apply. The bank was once moving dramatically; now it\'s stabilized by engineering. Predictions for protected reaches should be interpreted with extra caution.',
        },
      ],
    },
    {
      id: 'algorithm-encyclopedia',
      heading: 'Algorithm Quick Reference',
      icon: 'BookOpen',
      blocks: [
        {
          type: 'text',
          content:
            'This application uses 11 algorithms. Click any card below to learn more about how it works, its strengths, weaknesses, and when to trust it.',
        },
        {
          type: 'algorithms',
          algorithms: ALL_ALGORITHM_CARDS,
          expanded: false,
        },
      ],
    },
  ],
};

export default dashboard;
