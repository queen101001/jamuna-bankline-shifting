'use client';
import { useEffect, useRef, useState, useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
import {
  X, Terminal, CheckCircle, XCircle, Loader, Copy, Check,
  RefreshCw, Database,
} from 'lucide-react';
import { getTrainLogs, postTrain } from '@/lib/api';
import useAppStore from '@/store';

// ── Phase progress bar ────────────────────────────────────────────────────────

const PHASES = [
  { id: 'training',        label: 'Training model',            Icon: Loader,     spin: true  },
  { id: 'model_reloading', label: 'Reloading checkpoint',      Icon: RefreshCw,  spin: true  },
  { id: 'cache_building',  label: 'Pre-computing predictions', Icon: Database,   spin: false },
  { id: 'ready',           label: 'Ready — all years cached',  Icon: CheckCircle, spin: false },
];

function PhaseProgress({ phase }) {
  if (!phase || phase === 'failed') return null;
  const currentIdx = PHASES.findIndex((p) => p.id === phase);

  return (
    <div className="flex items-center gap-0 px-5 py-3 border-b overflow-x-auto" style={{ borderColor: 'var(--border)' }}>
      {PHASES.map((p, idx) => {
        const done = idx < currentIdx;
        const active = idx === currentIdx;
        const pending = idx > currentIdx;
        const color = done || active
          ? (p.id === 'ready' ? '#22c55e' : active ? 'var(--accent)' : '#22c55e')
          : 'var(--muted)';

        return (
          <div key={p.id} className="flex items-center">
            <div className="flex items-center gap-1.5 shrink-0">
              {done ? (
                <CheckCircle size={13} style={{ color: '#22c55e' }} />
              ) : (
                <p.Icon
                  size={13}
                  style={{ color }}
                  className={active && p.spin ? 'animate-spin' : ''}
                />
              )}
              <span
                className="text-xs font-medium whitespace-nowrap"
                style={{ color: pending ? 'var(--muted)' : color }}
              >
                {p.label}
              </span>
            </div>
            {idx < PHASES.length - 1 && (
              <div
                className="w-6 h-px mx-2 shrink-0"
                style={{ background: done ? '#22c55e' : 'var(--border)' }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Status badge ──────────────────────────────────────────────────────────────

function StatusBadge({ status, phase }) {
  if (!status || status === 'started') {
    return (
      <span className="flex items-center gap-1.5 text-xs px-2 py-0.5 rounded-full" style={{ background: 'rgba(6,182,212,0.12)', color: 'var(--accent)' }}>
        <Loader size={11} className="animate-spin" />
        Queued
      </span>
    );
  }
  if (status === 'running') {
    const label =
      phase === 'model_reloading' ? 'Reloading…' :
      phase === 'cache_building'  ? 'Caching…'   : 'Training…';
    return (
      <span className="flex items-center gap-1.5 text-xs px-2 py-0.5 rounded-full" style={{ background: 'rgba(234,179,8,0.12)', color: '#eab308' }}>
        <Loader size={11} className="animate-spin" />
        {label}
      </span>
    );
  }
  if (status.startsWith('completed')) {
    return (
      <span className="flex items-center gap-1.5 text-xs px-2 py-0.5 rounded-full" style={{ background: 'rgba(34,197,94,0.12)', color: '#22c55e' }}>
        <CheckCircle size={11} />
        Completed
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1.5 text-xs px-2 py-0.5 rounded-full" style={{ background: 'rgba(239,68,68,0.12)', color: '#ef4444' }}>
      <XCircle size={11} />
      Failed
    </span>
  );
}

// ── Log line ──────────────────────────────────────────────────────────────────

function LogLine({ line }) {
  let color = 'var(--text-dim)';
  if (line.includes('| ERROR') || line.includes('[ERROR]')) color = '#ef4444';
  else if (line.includes('| WARNING') || line.includes('[WARN]')) color = '#eab308';
  else if (line.includes('| SUCCESS') || line.includes('complete') || line.includes('Epoch')) color = '#22c55e';
  else if (line.includes('| INFO') || line.startsWith('[')) color = 'var(--text)';

  return (
    <div className="font-mono text-xs leading-relaxed whitespace-pre-wrap break-all" style={{ color }}>
      {line}
    </div>
  );
}

// ── Main drawer ───────────────────────────────────────────────────────────────

export default function TrainingDrawer() {
  const { isTrainingDrawerOpen, setTrainingDrawerOpen, trainingJobId, setTrainingJobId, setToastMessage } = useAppStore();
  const queryClient = useQueryClient();
  const router = useRouter();

  const [logs, setLogs] = useState([]);
  const [status, setStatus] = useState(null);
  const [phase, setPhase] = useState(null);
  const [isStarting, setIsStarting] = useState(false);
  const [startError, setStartError] = useState(null);
  const [copied, setCopied] = useState(false);
  const [toastVisible, setToastVisible] = useState(false);

  const logEndRef = useRef(null);
  const sinceRef = useRef(0);
  const pollRef = useRef(null);
  const prevPhaseRef = useRef(null);

  const isDone = status && (status.startsWith('completed') || status.startsWith('failed'));
  const isReady = phase === 'ready' && status?.startsWith('completed');

  // Auto-scroll
  useEffect(() => {
    if (isTrainingDrawerOpen) {
      logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, isTrainingDrawerOpen]);

  // Poll for logs + phase
  const pollLogs = useCallback(async (jobId) => {
    try {
      const data = await getTrainLogs(jobId, sinceRef.current);
      if (data.logs?.length) {
        setLogs((prev) => [...prev, ...data.logs]);
        sinceRef.current = data.total;
      }
      setStatus(data.status);
      if (data.phase) setPhase(data.phase);

      // Detect transition to ready — show toast and invalidate predictions
      if (data.phase === 'ready' && prevPhaseRef.current !== 'ready') {
        setToastVisible(true);
        setToastMessage({
          title: 'Model trained and cache ready',
          body: 'Predictions for 2021–2040 are now instant.',
        });
        queryClient.invalidateQueries({ queryKey: ['year-predictions'] });
        queryClient.invalidateQueries({ queryKey: ['health'] });
      }
      prevPhaseRef.current = data.phase;

      if (data.status.startsWith('completed') || data.status.startsWith('failed')) {
        clearInterval(pollRef.current);
      }
    } catch {
      // Backend may be busy during model reload; keep polling
    }
  }, [queryClient]);

  useEffect(() => {
    if (!trainingJobId || !isTrainingDrawerOpen) return;
    clearInterval(pollRef.current);
    pollRef.current = setInterval(() => pollLogs(trainingJobId), 2000);
    pollLogs(trainingJobId);
    return () => clearInterval(pollRef.current);
  }, [trainingJobId, isTrainingDrawerOpen, pollLogs]);

  // Reset state when drawer opens without a job
  useEffect(() => {
    if (isTrainingDrawerOpen && !trainingJobId) {
      setLogs([]);
      setStatus(null);
      setPhase(null);
      sinceRef.current = 0;
      setStartError(null);
      setToastVisible(false);
      prevPhaseRef.current = null;
    }
  }, [isTrainingDrawerOpen, trainingJobId]);

  async function handleStartTraining() {
    setIsStarting(true);
    setStartError(null);
    setLogs([]);
    setStatus('started');
    setPhase('training');
    sinceRef.current = 0;
    prevPhaseRef.current = null;
    try {
      const res = await postTrain({});
      setTrainingJobId(res.job_id);
    } catch (e) {
      setStartError(e.message || 'Failed to start training');
      setStatus(null);
      setPhase(null);
    } finally {
      setIsStarting(false);
    }
  }

  function handleClose() {
    setTrainingDrawerOpen(false);
    clearInterval(pollRef.current);
  }

  function handleCopyLogs() {
    navigator.clipboard.writeText(logs.join('\n')).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  function handleGoToDashboard() {
    handleClose();
    router.push('/');
  }

  function handleNewTraining() {
    setTrainingJobId(null);
    setLogs([]);
    setStatus(null);
    setPhase(null);
    sinceRef.current = 0;
    setStartError(null);
    setToastVisible(false);
    prevPhaseRef.current = null;
  }

  if (!isTrainingDrawerOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40"
        style={{ background: 'rgba(2,8,23,0.7)' }}
        onClick={handleClose}
      />

      {/* Drawer */}
      <div
        className="fixed bottom-0 left-0 right-0 z-50 flex flex-col rounded-t-2xl border-t"
        style={{
          background: 'var(--surface)',
          borderColor: 'var(--border)',
          maxHeight: '70vh',
          boxShadow: '0 -8px 32px rgba(0,0,0,0.5)',
        }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-5 py-4 border-b flex-shrink-0"
          style={{ borderColor: 'var(--border)' }}
        >
          <div className="flex items-center gap-3">
            <Terminal size={18} style={{ color: 'var(--accent)' }} />
            <div>
              <p className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
                Model Training
              </p>
              {trainingJobId && (
                <p className="text-xs font-mono" style={{ color: 'var(--muted)' }}>
                  job: {trainingJobId.slice(0, 8)}
                </p>
              )}
            </div>
            {status && <StatusBadge status={status} phase={phase} />}
          </div>

          <div className="flex items-center gap-2">
            {logs.length > 0 && (
              <button
                onClick={handleCopyLogs}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-colors"
                style={{ background: 'var(--card)', color: copied ? '#22c55e' : 'var(--text-dim)' }}
              >
                {copied ? <Check size={13} /> : <Copy size={13} />}
                {copied ? 'Copied' : 'Copy logs'}
              </button>
            )}
            <button
              onClick={handleClose}
              className="p-1.5 rounded-lg transition-colors"
              style={{ color: 'var(--text-dim)' }}
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Phase progress bar (shown while a job is running) */}
        {phase && <PhaseProgress phase={phase} />}

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-4 min-h-0">
          {/* No job started yet */}
          {!trainingJobId && !isStarting && (
            <div className="flex flex-col items-center justify-center h-full gap-4 py-10">
              <Terminal size={36} style={{ color: 'var(--muted)' }} />
              <div className="text-center">
                <p className="text-sm font-medium mb-1" style={{ color: 'var(--text)' }}>
                  Start Model Training
                </p>
                <p className="text-xs max-w-xs" style={{ color: 'var(--text-dim)' }}>
                  Trains the TFT model on all 100 series. After training, the model will
                  auto-reload and pre-compute predictions for 2021–2040 — all year queries
                  will then be instant.
                </p>
              </div>
              {startError && (
                <p className="text-xs text-center px-4" style={{ color: '#ef4444' }}>
                  {startError}
                </p>
              )}
              <button
                onClick={handleStartTraining}
                className="px-6 py-2.5 rounded-xl text-sm font-semibold transition-all"
                style={{ background: 'var(--accent)', color: '#020817' }}
              >
                Start Training
              </button>
            </div>
          )}

          {/* Starting spinner */}
          {isStarting && (
            <div className="flex items-center gap-2 py-4" style={{ color: 'var(--text-dim)' }}>
              <Loader size={14} className="animate-spin" />
              <span className="text-xs">Submitting training job…</span>
            </div>
          )}

          {/* Log output */}
          {logs.length > 0 && (
            <div className="space-y-0.5">
              {logs.map((line, i) => (
                <LogLine key={i} line={line} />
              ))}
              <div ref={logEndRef} />
            </div>
          )}

          {/* Waiting for first log */}
          {trainingJobId && logs.length === 0 && !isStarting && (
            <div className="flex items-center gap-2 py-4" style={{ color: 'var(--text-dim)' }}>
              <Loader size={14} className="animate-spin" />
              <span className="text-xs font-mono">Waiting for logs…</span>
            </div>
          )}

          {/* Ready banner */}
          {isReady && (
            <div
              className="mt-4 rounded-lg px-4 py-3 flex items-center justify-between gap-4"
              style={{
                background: 'rgba(34,197,94,0.08)',
                border: '1px solid rgba(34,197,94,0.25)',
              }}
            >
              <div className="flex items-center gap-2">
                <CheckCircle size={15} className="shrink-0" style={{ color: '#22c55e' }} />
                <p className="text-xs font-medium" style={{ color: '#22c55e' }}>
                  Model trained and cache ready. Predictions for 2021–2040 are now instant.
                </p>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <button
                  onClick={handleNewTraining}
                  className="text-xs font-medium px-3 py-1.5 rounded-lg transition-colors"
                  style={{ background: 'var(--card)', color: 'var(--text-dim)' }}
                >
                  Retrain
                </button>
                <button
                  onClick={handleGoToDashboard}
                  className="text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors"
                  style={{ background: 'rgba(34,197,94,0.15)', color: '#22c55e' }}
                >
                  View Dashboard →
                </button>
              </div>
            </div>
          )}

          {/* Failed banner */}
          {isDone && status.startsWith('failed') && (
            <div className="mt-4 space-y-2">
              <div
                className="rounded-lg px-4 py-3 text-xs font-mono"
                style={{
                  background: 'rgba(239,68,68,0.08)',
                  color: '#ef4444',
                  border: '1px solid rgba(239,68,68,0.2)',
                }}
              >
                Training failed. Check logs above for details.
              </div>
              <button
                onClick={handleNewTraining}
                className="text-xs font-semibold px-4 py-2 rounded-lg transition-colors"
                style={{ background: 'var(--card)', color: 'var(--text)' }}
              >
                Try Again
              </button>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
