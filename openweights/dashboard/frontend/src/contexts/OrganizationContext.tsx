import { createContext, useContext, useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Organization } from '../types';
import { api } from '../api';
import { useAuth } from './AuthContext';

interface OrganizationContextType {
  organizations: Organization[];
  currentOrganization: Organization | null;
  setCurrentOrganization: (org: Organization) => void;
  loadOrganizations: () => Promise<void>;
  loading: boolean;
}

const OrganizationContext = createContext<OrganizationContextType | null>(null);

export function OrganizationProvider({ children }: { children: React.ReactNode }) {
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [currentOrganization, setCurrentOrganization] = useState<Organization | null>(null);
  const [loading, setLoading] = useState(true);
  const [initialLoaded, setInitialLoaded] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();

  const loadOrganizations = async () => {
    try {
      setLoading(true);
      const orgs = await api.getOrganizations();
      console.log('Loaded organizations:', orgs);
      setOrganizations(orgs);
      setLoading(false);
      setInitialLoaded(true);
    } catch (error) {
      console.error('Failed to load organizations:', error);
      setLoading(false);
      setInitialLoaded(true);
    }
  };

  // Load organizations when user changes
  useEffect(() => {
    if (user && !initialLoaded) {
      loadOrganizations();
    } else if (!user) {
      setOrganizations([]);
      setCurrentOrganization(null);
      setInitialLoaded(false);
    }
  }, [user, initialLoaded]);

  // Try to extract organization ID from URL and set it as current
  useEffect(() => {
    const match = location.pathname.match(/^\/([^/]+)/);
    if (match && match[1] !== 'organizations' && match[1] !== 'login') {
      const orgId = match[1];
      console.log('Found organization ID in URL:', orgId);
      const org = organizations.find(o => o.id === orgId);
      if (org && (!currentOrganization || currentOrganization.id !== org.id)) {
        console.log('Setting current organization from URL:', org);
        setCurrentOrganization(org);
      }
    }
  }, [location.pathname, organizations]);

  const handleSetOrganization = (org: Organization) => {
    console.log('Setting organization:', org);
    setCurrentOrganization(org);
    // If we're on the organizations page, navigate to jobs
    if (location.pathname === '/organizations') {
      navigate(`/${org.id}/jobs`);
    }
  };

  return (
    <OrganizationContext.Provider
      value={{
        organizations,
        currentOrganization,
        setCurrentOrganization: handleSetOrganization,
        loadOrganizations,
        loading
      }}
    >
      {children}
    </OrganizationContext.Provider>
  );
}

export const useOrganization = () => {
  const context = useContext(OrganizationContext);
  if (!context) {
    throw new Error('useOrganization must be used within an OrganizationProvider');
  }
  return context;
};
