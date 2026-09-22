/// <reference types="vite/client" />

interface Window {
  __OPENWEIGHTS_CONFIG__?: {
    supabaseUrl: string;
    supabaseAnonKey: string;
  };
}
