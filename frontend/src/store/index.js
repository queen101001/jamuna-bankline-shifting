'use client';
import { create } from 'zustand';

const useAppStore = create((set) => ({
  selectedYear: 2021,
  setSelectedYear: (year) => set({ selectedYear: year }),

  selectedReach: null,
  setSelectedReach: (reach) => set({ selectedReach: reach }),

  // Training drawer state
  trainingJobId: null,
  setTrainingJobId: (id) => set({ trainingJobId: id }),

  isTrainingDrawerOpen: false,
  setTrainingDrawerOpen: (open) => set({ isTrainingDrawerOpen: open }),

  // Toast notification: { title, body } or null
  toastMessage: null,
  setToastMessage: (msg) => set({ toastMessage: msg }),
}));

export default useAppStore;
