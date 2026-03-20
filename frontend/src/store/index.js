'use client';
import { create } from 'zustand';

const useAppStore = create((set) => ({
  selectedYear: 2021,
  confirmedYear: 2021,
  setSelectedYear: (year) => set({ selectedYear: year }),
  confirmYear: () => set((s) => ({ confirmedYear: s.selectedYear })),

  selectedReach: null,
  setSelectedReach: (reach) => set({ selectedReach: reach }),

  // Active algorithm for predictions (TFT or a baseline name)
  activeAlgorithm: 'tft',
  setActiveAlgorithm: (algo) => set({ activeAlgorithm: algo }),

  // Training drawer state
  trainingJobId: null,
  setTrainingJobId: (id) => set({ trainingJobId: id }),

  isTrainingDrawerOpen: false,
  setTrainingDrawerOpen: (open) => set({ isTrainingDrawerOpen: open }),

  // Toast notification: { title, body } or null
  toastMessage: null,
  setToastMessage: (msg) => set({ toastMessage: msg }),

  // Validation state
  validationData: null,
  setValidationData: (data) => set({ validationData: data }),
  validationResults: null,
  setValidationResults: (r) => set({ validationResults: r }),
  clearValidation: () => set({ validationData: null, validationResults: null }),
}));

export default useAppStore;
