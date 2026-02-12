import YearSelector from '@/components/dashboard/YearSelector';
import ReachGrid from '@/components/dashboard/ReachGrid';

export default function DashboardPage() {
  return (
    <div className="min-h-screen px-4 py-8 max-w-[1600px] mx-auto">
      {/* Hero header */}
      <div className="mb-10 text-center">
        <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text)' }}>
          Jamuna River Bankline Prediction
        </h1>
        <p className="text-sm max-w-xl mx-auto" style={{ color: 'var(--text-dim)' }}>
          Select a forecast year to view predicted bankline positions for all 50 reaches.
          Left bank positive = erosion · Right bank negative = erosion.
        </p>
      </div>

      {/* Year selector */}
      <div className="flex justify-center mb-10">
        <YearSelector />
      </div>

      {/* Reach grid */}
      <ReachGrid />
    </div>
  );
}
