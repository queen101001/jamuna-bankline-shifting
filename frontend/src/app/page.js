'use client';
import useAppStore from '@/store';
import YearSelector from '@/components/dashboard/YearSelector';
import AlgorithmSelector from '@/components/ui/AlgorithmSelector';
import ReachGrid from '@/components/dashboard/ReachGrid';
import DownloadPanel from '@/components/dashboard/DownloadPanel';
import InfoButton from '@/components/ui/InfoButton';

export default function DashboardPage() {
  const confirmedYear = useAppStore((s) => s.confirmedYear);
  const isForecast = confirmedYear > 2020;

  return (
    <div className="min-h-screen px-4 py-8 max-w-[1600px] mx-auto">
      {/* Hero header */}
      <div className="mb-10 text-center">
        <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text)' }}>
          Jamuna River Bankline Prediction
        </h1>
        <p className="text-sm max-w-xl mx-auto" style={{ color: 'var(--text-dim)' }}>
          Select a forecast year and algorithm to view predicted bankline positions for all 50 reaches.
          Negative = erosion · Positive = deposition (both banks).
        </p>
      </div>

      {/* Year + Algorithm selectors */}
      <div className="flex flex-wrap justify-center items-start gap-8 mb-10">
        <YearSelector />
        {isForecast && <AlgorithmSelector />}
      </div>

      {/* Download panel (forecast years only) */}
      {isForecast && (
        <div className="flex justify-center mb-8">
          <DownloadPanel />
        </div>
      )}

      {/* Reach grid */}
      <ReachGrid />

      <InfoButton pageId="dashboard" />
    </div>
  );
}
