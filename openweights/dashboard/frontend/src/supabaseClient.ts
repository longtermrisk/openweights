import { createClient } from '@supabase/supabase-js'
import { Database } from './types/supabase'

const supabaseUrl = window.__OPENWEIGHTS_CONFIG__?.supabaseUrl || import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = window.__OPENWEIGHTS_CONFIG__?.supabaseAnonKey || import.meta.env.VITE_SUPABASE_ANON_KEY

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables')
}

export const supabase = createClient<Database>(supabaseUrl, supabaseAnonKey)
