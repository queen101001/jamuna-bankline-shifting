'use client';
import useAppStore from '@/store';
import ExcelUploader from '@/components/validate/ExcelUploader';
import ValidationResults from '@/components/validate/ValidationResults';

export default function ValidatePage() {
  const { validationData } = useAppStore();

  return (
    <main className="min-h-screen pt-24 px-6 pb-12" style={{ background: 'var(--bg)' }}>
      <div className="max-w-7xl mx-auto flex flex-col gap-6">
        <div>
          <h1 className="text-3xl font-bold" style={{ color: 'var(--text)' }}>
            Model Validation
          </h1>
          <p className="text-sm mt-1" style={{ color: 'var(--muted)' }}>
            Upload an Excel file with observed bankline distances to compare against TFT predictions.
          </p>
        </div>

        {validationData ? <ValidationResults /> : <ExcelUploader />}
      </div>
    </main>
  );
}
