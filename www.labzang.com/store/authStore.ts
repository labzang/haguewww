/**
 * 인증 상태 관리 스토어 (zustand)
 * Access Token은 메모리에만 저장 (persist 제외)
 */

import { create } from 'zustand';

interface AuthState {
  accessToken: string | null;
}

interface AuthActions {
  setAccessToken: (token: string | null) => void;
  getAccessToken: () => string | null;
  clearAccessToken: () => void;
}

type AuthStore = AuthState & AuthActions;

export const useAuthStore = create<AuthStore>((set, get) => ({
  // State
  accessToken: null, // 메모리에만 저장 (짧은 수명: 5~15분)

  // Actions
  setAccessToken: (token) => {
    set({ accessToken: token });
  },

  getAccessToken: () => {
    return get().accessToken;
  },

  clearAccessToken: () => {
    set({ accessToken: null });
  },
}));

