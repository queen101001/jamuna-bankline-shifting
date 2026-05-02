'use client';
import { useState } from 'react';
import useAppStore from '@/store';
import ExcelUploader from '@/components/validate/ExcelUploader';
import ModelEvaluationTable from '@/components/validate/ModelEvaluationTable';
import ValidationResults from '@/components/validate/ValidationResults';
import InfoButton from '@/components/ui/InfoButton';

const VALIDATE_TABS = [
  { key: 'validation', label: 'Model Validation' },
  { key: 'evaluation', label: 'Model Evaluation' },
];

export default function ValidatePage() {
  const { validationData } = useAppStore();
  const [activeValidateTab, setActiveValidateTab] = useState('validation');

  return (
    <main className="min-h-screen pt-24 px-6 pb-12" style={{ background: 'var(--bg)' }}>
      <div className="max-w-7xl mx-auto flex flex-col gap-6">
        <div>
          <h1 className="text-3xl font-bold" style={{ color: 'var(--text)' }}>
            Model Validation
          </h1>
          <p className="text-sm mt-1" style={{ color: 'var(--muted)' }}>
            Upload an Excel file with observed bankline distances to compare against all 11 algorithms.
          </p>
        </div>

        {validationData && (
          <div
            className="flex w-fit rounded-xl border p-1"
            style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}
          >
            {VALIDATE_TABS.map((tab) => {
              const active = activeValidateTab === tab.key;

              return (
                <button
                  key={tab.key}
                  type="button"
                  onClick={() => setActiveValidateTab(tab.key)}
                  className="rounded-lg px-4 py-2 text-sm font-medium transition-colors"
                  style={{
                    background: active ? 'rgba(6,182,212,0.16)' : 'transparent',
                    color: active ? 'var(--accent)' : 'var(--text-dim)',
                  }}
                >
                  {tab.label}
                </button>
              );
            })}
          </div>
        )}

        {!validationData ? (
          <ExcelUploader />
        ) : activeValidateTab === 'validation' ? (
          <ValidationResults />
        ) : (
          <ModelEvaluationTable validationData={validationData} />
        )}
      </div>

      <InfoButton pageId="validate" />
    </main>
  );
}
