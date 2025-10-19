import { createContext, useContext, useEffect, useState } from 'react';
import { Session, User } from '@supabase/supabase-js';
import { supabase } from '../supabaseClient';
import axios from 'axios';

const API_URL = import.meta.env.PROD ? '' : 'http://localhost:8124';

interface AuthContextType {
  session: Session | null;
  user: User | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<{ error: Error | null }>;
  signInWithApiKey: (apiKey: string) => Promise<{ error: Error | null }>;
  signUp: (email: string, password: string) => Promise<{ error: Error | null }>;
  signOut: () => Promise<void>;
  resetPassword: (email: string) => Promise<{ error: Error | null }>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check for stored JWT from API key login
    const storedJwt = localStorage.getItem('openweights_jwt');
    if (storedJwt) {
      // Recreate the mock session
      const mockSession = {
        access_token: storedJwt,
        refresh_token: storedJwt,
        expires_in: 3600,
        expires_at: Math.floor(Date.now() / 1000) + 3600,
        token_type: 'bearer',
        user: {
          id: 'api-key-user',
          email: 'api-key@openweights.com',
          aud: 'authenticated',
          role: 'authenticated',
          created_at: new Date().toISOString(),
          app_metadata: {},
          user_metadata: {},
        }
      } as Session;

      setSession(mockSession);
      setUser(mockSession.user);
      setLoading(false);
    } else {
      // Get initial session from Supabase Auth
      supabase.auth.getSession().then(({ data: { session } }) => {
        setSession(session);
        setUser(session?.user ?? null);
        setLoading(false);
      });
    }

    // Listen for auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      // Only update if not using API key auth
      if (!localStorage.getItem('openweights_jwt')) {
        setSession(session);
        setUser(session?.user ?? null);
        setLoading(false);
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  const signIn = async (email: string, password: string) => {
    try {
      const { error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      return { error };
    } catch (error) {
      return { error: error as Error };
    }
  };

  const signInWithApiKey = async (apiKey: string) => {
    try {
      // Exchange API key for JWT
      const response = await axios.post(`${API_URL}/auth/exchange-api-key`, {
        api_key: apiKey
      });

      const jwt = response.data.jwt;

      // Store the JWT in localStorage for API calls
      localStorage.setItem('openweights_jwt', jwt);

      // Create a mock session for the auth context
      // This JWT is for database access, not Supabase Auth
      const mockSession = {
        access_token: jwt,
        refresh_token: jwt,
        expires_in: 3600,
        expires_at: Math.floor(Date.now() / 1000) + 3600,
        token_type: 'bearer',
        user: {
          id: 'api-key-user',
          email: 'api-key@openweights.com',
          aud: 'authenticated',
          role: 'authenticated',
          created_at: new Date().toISOString(),
          app_metadata: {},
          user_metadata: {},
        }
      } as Session;

      setSession(mockSession);
      setUser(mockSession.user);

      return { error: null };
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const message = error.response?.data?.detail || error.message;
        return { error: new Error(message) };
      }
      return { error: error as Error };
    }
  };

  const signUp = async (email: string, password: string) => {
    try {
      const { error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          emailRedirectTo: window.location.origin,
        },
      });
      return { error };
    } catch (error) {
      return { error: error as Error };
    }
  };

  const signOut = async () => {
    // Clear API key JWT if present
    localStorage.removeItem('openweights_jwt');

    // Also sign out from Supabase Auth
    await supabase.auth.signOut();

    // Clear local state
    setSession(null);
    setUser(null);
  };

  const resetPassword = async (email: string) => {
    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/reset-password`,
      });
      return { error };
    } catch (error) {
      return { error: error as Error };
    }
  };

  const value = {
    session,
    user,
    loading,
    signIn,
    signInWithApiKey,
    signUp,
    signOut,
    resetPassword,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
