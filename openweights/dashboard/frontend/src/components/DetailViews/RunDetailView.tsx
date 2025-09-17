import React, { useEffect, useState, useCallback, Suspense, lazy } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
    Paper,
    Typography,
    Box,
    Chip,
    FormControlLabel,
    Switch,
    CircularProgress,
    Button,
    Collapse
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import { RunWithJobAndWorker } from '../../types';
import { api } from '../../api';
import { RefreshButton } from '../RefreshButton';
import { useOrganization } from '../../contexts/OrganizationContext';

// Lazy load components
const MetricsPlots = lazy(() => import('./MetricsPlots').then(module => ({ default: module.MetricsPlots })));
const LogProbVisualization = lazy(() => import('../LogProbVisualization').then(module => ({ default: module.LogProbVisualization })));

// Loading placeholder component
const LoadingPlaceholder = () => (
    <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
    </Box>
);

// Section button component for consistent styling
const SectionButton: React.FC<{
    onClick: () => void;
    expanded: boolean;
    children: React.ReactNode;
}> = ({ onClick, expanded, children }) => (
    <Button
        onClick={onClick}
        endIcon={expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        fullWidth
        sx={{
            justifyContent: 'flex-start',
            backgroundColor: expanded ? 'rgba(0, 0, 0, 0.04)' : 'transparent',
            '&:hover': {
                backgroundColor: expanded ? 'rgba(0, 0, 0, 0.08)' : 'rgba(0, 0, 0, 0.04)'
            }
        }}
    >
        {children}
    </Button>
);

export const RunDetailView: React.FC = () => {
    const { orgId, runId } = useParams<{ orgId: string; runId: string }>();
    const { currentOrganization } = useOrganization();
    const [run, setRun] = useState<RunWithJobAndWorker | null>(null);
    const [logContent, setLogContent] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [lastRefresh, setLastRefresh] = useState<Date>();
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [events, setEvents] = useState<any[]>([]);

    // UI state for collapsible sections
    const [showLogProbs, setShowLogProbs] = useState(false);
    const [showMetrics, setShowMetrics] = useState(false);
    const [showLogs, setShowLogs] = useState(false);

    // Pagination for logs
    const [logPage, setLogPage] = useState(1);
    const logsPerPage = 1000;

    const AUTO_REFRESH_INTERVAL = 10000;

    const fetchRun = useCallback(async () => {
        if (!orgId || !runId) return;
        setLoading(true);
        try {
            const data = await api.getRun(orgId, runId);
            setRun(data);
            setLastRefresh(new Date());
        } catch (error) {
            console.error('Error in fetchRun:', error);
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            setError(errorMessage);
        } finally {
            setLoading(false);
        }
    }, [orgId, runId]);

    const fetchLogs = useCallback(async () => {
        if (!orgId || !runId) return;
        try {
            const logs = await api.getRunLogs(orgId, runId);
            setLogContent(logs);
        } catch (error) {
            console.error('Error fetching logs:', error);
        }
    }, [orgId, runId]);

    const fetchEvents = useCallback(async () => {
        if (!orgId || !runId) return;
        try {
            const runEvents = await api.getRunEvents(orgId, runId);
            setEvents(runEvents);
        } catch (error) {
            console.error('Error fetching events:', error);
        }
    }, [orgId, runId]);

    // Initial load and background prefetch
    useEffect(() => {
        const loadAllData = async () => {
            await fetchRun();
            Promise.all([
                fetchLogs(),
                fetchEvents()
            ]).catch(error => {
                console.error('Error in background data loading:', error);
            });
        };
        loadAllData();
    }, [fetchRun, fetchLogs, fetchEvents]);

    // Auto-refresh effect
    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (autoRefresh && run?.status === 'in_progress') {
            interval = setInterval(() => {
                Promise.all([
                    fetchRun(),
                    fetchLogs(),
                    fetchEvents()
                ]).catch(error => {
                    console.error('Error in auto-refresh:', error);
                });
            }, AUTO_REFRESH_INTERVAL);
        }
        return () => {
            if (interval) {
                clearInterval(interval);
            }
        };
    }, [autoRefresh, fetchRun, fetchLogs, fetchEvents, run?.status]);

    // Filter logprob events and extract the data field
    const logprobEvents = events
        .filter(event => event.data?.type === 'logprobs')
        .map(event => event.data);

    if (!orgId || !currentOrganization || !run) {
        return <Typography>Loading...</Typography>;
    }

    if (!runId) {
        return <Typography>Run ID is required</Typography>;
    }

    // Paginate logs
    const paginatedLogs = logContent
        ? logContent
            .split('\n')
            .slice((logPage - 1) * logsPerPage, logPage * logsPerPage)
            .join('\n')
        : '';
    const totalLogPages = logContent
        ? Math.ceil(logContent.split('\n').length / logsPerPage)
        : 0;

    return (
        <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <Typography variant="h4" sx={{ flexGrow: 1 }}>Run: {run.id}</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={autoRefresh}
                                onChange={(e) => setAutoRefresh(e.target.checked)}
                                name="autoRefresh"
                            />
                        }
                        label="Auto-refresh"
                    />
                    <RefreshButton
                        onRefresh={fetchRun}
                        loading={loading}
                        lastRefresh={lastRefresh}
                    />
                </Box>
            </Box>

            <Box sx={{ mb: 3 }}>
                <Chip label={`Status: ${run.status}`} sx={{ mr: 1 }} />
                <Chip
                    label={`Job: ${run.job_id}`}
                    component={Link}
                    to={`/${orgId}/jobs/${run.job_id}`}
                    clickable
                    sx={{ mr: 1 }}
                />
                {run.worker && (
                    <Chip
                        label={`Worker: ${run.worker_id}`}
                        component={Link}
                        to={`/${orgId}/workers/${run.worker_id}`}
                        clickable
                        sx={{ mr: 1 }}
                    />
                )}
            </Box>

            {error && (
                <Box sx={{ mb: 3 }}>
                    <Typography color="error">Error: {error}</Typography>
                </Box>
            )}

            {/* Main content sections */}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {/* Log Probabilities Section */}
                {logprobEvents.length > 0 && (
                    <Box>
                        <SectionButton
                            onClick={() => setShowLogProbs(!showLogProbs)}
                            expanded={showLogProbs}
                        >
                            Log Probabilities ({logprobEvents.length} events)
                        </SectionButton>
                        <Collapse in={showLogProbs}>
                            <Box sx={{ mt: 2 }}>
                                <Suspense fallback={<LoadingPlaceholder />}>
                                    <LogProbVisualization
                                        events={logprobEvents}
                                        getFileContent={(fileId: string) => api.getFileContent(orgId, fileId)}
                                    />
                                </Suspense>
                            </Box>
                        </Collapse>
                    </Box>
                )}

                {/* Metrics Section */}
                <Box>
                    <SectionButton
                        onClick={() => setShowMetrics(!showMetrics)}
                        expanded={showMetrics}
                    >
                        Training Metrics
                    </SectionButton>
                    <Collapse in={showMetrics}>
                        <Box sx={{ mt: 2 }}>
                            <Suspense fallback={<LoadingPlaceholder />}>
                                <MetricsPlots orgId={orgId} runId={runId} />
                            </Suspense>
                        </Box>
                    </Collapse>
                </Box>

                {/* Logs Section */}
                <Box>
                    <SectionButton
                        onClick={() => setShowLogs(!showLogs)}
                        expanded={showLogs}
                    >
                        Log Output
                    </SectionButton>
                    <Collapse in={showLogs}>
                        {logContent && (
                            <Box sx={{ mt: 2 }}>
                                <Paper
                                    sx={{
                                        p: 2,
                                        bgcolor: 'grey.100',
                                        maxHeight: '500px',
                                        overflow: 'auto'
                                    }}
                                >
                                    <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
                                        {paginatedLogs}
                                    </pre>
                                </Paper>
                                {totalLogPages > 1 && (
                                    <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center', gap: 2 }}>
                                        <Button
                                            disabled={logPage === 1}
                                            onClick={() => setLogPage(p => Math.max(1, p - 1))}
                                        >
                                            Previous
                                        </Button>
                                        <Typography>
                                            Page {logPage} of {totalLogPages}
                                        </Typography>
                                        <Button
                                            disabled={logPage === totalLogPages}
                                            onClick={() => setLogPage(p => Math.min(totalLogPages, p + 1))}
                                        >
                                            Next
                                        </Button>
                                    </Box>
                                )}
                            </Box>
                        )}
                    </Collapse>
                </Box>
            </Box>
        </Paper>
    );
};
