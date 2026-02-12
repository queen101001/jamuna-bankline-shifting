'use client';
import { create } from 'zustand';

const useAppStore = create((set) => ({
  selectedYear: 2021,
  setSelectedYear: (year) => set({ selectedYear: year }),

  selectedReach: null,
  setSelectedReach: (reach) => set({ selectedReach: reach }),
}));

export default useAppStore;
