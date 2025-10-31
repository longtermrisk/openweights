import { BrowserRouter as Router, Routes, Route, Link, Navigate, useNavigate, useParams, useLocation } from 'react-router-dom';
import {
    AppBar,
    Toolbar,
    Typography,
    Container,
    Box,
    Button,
    Menu,
    MenuItem,
    IconButton
} from '@mui/material';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import SettingsIcon from '@mui/icons-material/Settings';
import { JobsView } from './components/JobsView';
import { WorkersView } from './components/WorkersView';
import { JobDetailView, RunDetailView, WorkerDetailView } from './components/DetailViews';
import { Auth } from './components/Auth/Auth';
import { OrganizationList } from './components/Organizations/OrganizationList';
import { OrganizationDetail } from './components/Organizations/OrganizationDetail';
import { OrganizationSwitcher } from './components/Organizations/OrganizationSwitcher';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { OrganizationProvider, useOrganization } from './contexts/OrganizationContext';
import { useState, useEffect } from 'react';

function PrivateRoute({ children }: { children: React.ReactNode }) {
    const { user, loading } = useAuth();

    if (loading) {
        return <Typography>Loading...</Typography>;
    }

    if (!user) {
        return <Navigate to="/login" />;
    }

    return <>{children}</>;
}

function NavBar() {
    const { user, signOut } = useAuth();
    const { currentOrganization } = useOrganization();
    const navigate = useNavigate();
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
    const open = Boolean(anchorEl);

    const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
        setAnchorEl(event.currentTarget);
    };

    const handleMenuClose = () => {
        setAnchorEl(null);
    };

    const handleMenuItemClick = (action: string) => {
        handleMenuClose();
        if (action === 'logout') {
            signOut();
        } else if (action === 'settings' && currentOrganization) {
            navigate(`/${currentOrganization.id}/settings`);
        }
    };

    if (!user) return null;

    return (
        <AppBar position="static">
            <Toolbar variant="dense">
                <img
                    src="/ow.svg"
                    alt="OpenWeights Logo"
                    style={{
                        height: '24px',
                        width: '24px',
                        marginRight: '8px'
                    }}
                />
                <Typography variant="h6" component="div" sx={{ flexGrow: 0, fontSize: '0.95rem' }}>
                    OpenWeights
                </Typography>

                <OrganizationSwitcher />

                {currentOrganization && (
                    <>
                        <Button color="inherit" component={Link} to={`/${currentOrganization.id}/jobs`}>Jobs</Button>

                        <Button color="inherit" component={Link} to={`/${currentOrganization.id}/workers`}>Workers</Button>
                    </>
                )}

                <Box sx={{ flexGrow: 1 }} />

                {currentOrganization && (
                    <IconButton
                        color="inherit"
                        onClick={() => navigate(`/${currentOrganization.id}/settings`)}
                        sx={{ mr: 1 }}
                    >
                        <SettingsIcon />
                    </IconButton>
                )}

                <IconButton
                    color="inherit"
                    onClick={handleMenuClick}
                    aria-label="more"
                    aria-controls={open ? 'more-menu' : undefined}
                    aria-haspopup="true"
                    aria-expanded={open ? 'true' : undefined}
                >
                    <MoreVertIcon />
                </IconButton>
                <Menu
                    id="more-menu"
                    anchorEl={anchorEl}
                    open={open}
                    onClose={handleMenuClose}
                    MenuListProps={{
                        'aria-labelledby': 'more-button',
                    }}
                >
                    <MenuItem onClick={() => handleMenuItemClick('logout')}>Logout</MenuItem>
                </Menu>
            </Toolbar>
        </AppBar>
    );
}

function OrganizationRoutes() {
    const { orgId } = useParams();
    const { currentOrganization, organizations, setCurrentOrganization, loading } = useOrganization();
    const location = useLocation();
    const navigate = useNavigate();

    useEffect(() => {
        console.log('OrganizationRoutes effect:', {
            orgId,
            currentOrganization,
            organizations,
            loading,
            pathname: location.pathname
        });

        if (!loading && organizations.length > 0 && orgId) {
            const org = organizations.find(o => o.id === orgId);
            if (org) {
                if (!currentOrganization || currentOrganization.id !== orgId) {
                    console.log('Setting organization from route:', org);
                    setCurrentOrganization(org);
                }
            } else {
                console.log('Organization not found, redirecting');
                navigate('/organizations');
            }
        }
    }, [orgId, organizations, currentOrganization, loading, location.pathname]);

    if (loading) {
        return <Typography>Loading organizations...</Typography>;
    }

    if (!organizations.length) {
        return <Navigate to="/organizations" />;
    }

    if (orgId && !organizations.find(o => o.id === orgId)) {
        return <Navigate to="/organizations" />;
    }

    // Don't render routes until we have the correct organization set
    if (orgId && (!currentOrganization || currentOrganization.id !== orgId)) {
        return <Typography>Loading organization {orgId}...</Typography>;
    }

    return (
        <Routes>
            <Route path="jobs" element={<JobsView />} />
            <Route path="jobs/:jobId" element={<JobDetailView />} />

            <Route path="workers" element={<WorkersView />} />
            <Route path="workers/:workerId" element={<WorkerDetailView />} />
            <Route path="runs/:runId" element={<RunDetailView />} />
            <Route path="settings" element={<OrganizationDetail />} />
            <Route path="/" element={<Navigate to="jobs" />} />
        </Routes>
    );
}

function AppContent() {
    const { user } = useAuth();
    const location = useLocation();
    const { organizations, loading } = useOrganization();
    const navigate = useNavigate();

    useEffect(() => {
        console.log('AppContent effect:', {
            user,
            organizations,
            loading,
            pathname: location.pathname
        });

        if (!loading && user && organizations.length > 0) {
            // Only redirect from root or /organizations
            if (location.pathname === '/' || location.pathname === '/organizations') {
                // If user has only one organization, redirect to it
                if (organizations.length === 1) {
                    const org = organizations[0];
                    navigate(`/${org.id}/jobs`);
                }
            }
        }
    }, [user, organizations, loading, location.pathname]);

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', width: '100%' }}>
            <NavBar />
            <Box component="main" sx={{ flexGrow: 1, width: '100%', height: '100%', overflow: 'auto' }}>
                <Container maxWidth={false} sx={{ mt: 1.5, mb: 1.5, height: 'calc(100vh - 60px)' }}>
                    <Routes>
                        <Route path="/login" element={<Auth />} />
                        <Route path="/organizations" element={
                            <PrivateRoute>
                                <OrganizationList />
                            </PrivateRoute>
                        } />

                        {/* Organization-specific routes */}
                        <Route path="/:orgId/*" element={
                            <PrivateRoute>
                                <OrganizationRoutes />
                            </PrivateRoute>
                        } />

                        {/* Root redirect */}
                        <Route path="/" element={
                            <PrivateRoute>
                                <Navigate to="/organizations" />
                            </PrivateRoute>
                        } />
                    </Routes>
                </Container>
            </Box>
        </Box>
    );
}

function App() {
    return (
        <Router>
            <AuthProvider>
                <OrganizationProvider>
                    <AppContent />
                </OrganizationProvider>
            </AuthProvider>
        </Router>
    );
}

export default App;
