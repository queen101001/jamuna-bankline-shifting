export const EXPECTED_BANK_POINT_COUNT = 50;

const BANK_SECTIONS = [
  { key: 'left', label: 'Left Bank' },
  { key: 'right', label: 'Right Bank' },
];

export function computeEvaluationMetrics(pairs, expectedCount = EXPECTED_BANK_POINT_COUNT) {
  const count = pairs.length;

  if (count !== expectedCount) {
    return {
      complete: false,
      count,
      rmse: null,
      mae: null,
      r2: null,
    };
  }

  const observedMean = pairs.reduce((sum, pair) => sum + pair.observed, 0) / count;
  const squaredErrorSum = pairs.reduce((sum, pair) => {
    const error = pair.observed - pair.predicted;
    return sum + error * error;
  }, 0);
  const absoluteErrorSum = pairs.reduce(
    (sum, pair) => sum + Math.abs(pair.observed - pair.predicted),
    0,
  );
  const observedVarianceSum = pairs.reduce((sum, pair) => {
    const centered = pair.observed - observedMean;
    return sum + centered * centered;
  }, 0);

  return {
    complete: true,
    count,
    rmse: Math.sqrt(squaredErrorSum / count),
    mae: absoluteErrorSum / count,
    r2: observedVarianceSum > 0 ? 1 - squaredErrorSum / observedVarianceSum : null,
  };
}

export function buildPredictionMap(predictionResponse) {
  const map = new Map();

  for (const point of predictionResponse?.predictions ?? []) {
    map.set(`${point.reach_id}-${point.bank_side}`, point.q50);
  }

  return map;
}

export function buildEvaluationRows({
  years,
  observedData,
  predictionMapsByYear,
  expectedCount = EXPECTED_BANK_POINT_COUNT,
}) {
  return years.flatMap((year) =>
    BANK_SECTIONS.map((section) => {
      const predictionMap = predictionMapsByYear[year] ?? new Map();
      const observedRows = observedData
        .filter((row) => row.year === year && row.bank_side === section.key)
        .sort((a, b) => a.reach_id - b.reach_id);
      const pairs = observedRows
        .map((row) => {
          const predicted = predictionMap.get(`${row.reach_id}-${row.bank_side}`);
          if (predicted == null) return null;
          return {
            reach_id: row.reach_id,
            observed: row.observed,
            predicted,
          };
        })
        .filter(Boolean);

      return {
        year,
        bankSide: section.key,
        section: section.label,
        expectedCount,
        observedCount: observedRows.length,
        pairCount: pairs.length,
        metrics: computeEvaluationMetrics(pairs, expectedCount),
      };
    }),
  );
}
