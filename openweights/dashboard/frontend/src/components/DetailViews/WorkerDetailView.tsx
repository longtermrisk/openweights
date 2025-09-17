import React, { useEffect, useState, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
    Paper,
    Typography,
    Box,
    Chip,
    List,
    ListItem,
    ListItemText,
    FormControlLabel,
    Switch,
    Snackbar,
    Tabs,
    Tab,
    CircularProgress
} from '@mui/material';
import { LoadingButton } from '@mui/lab';
import PowerSettingsNewIcon from '@mui/icons-material/PowerSettingsNew';
import { WorkerWithRuns } from '../../types';
import { api } from '../../api';
import { RefreshButton } from '../RefreshButton';
import { useOrganization } from '../../contexts/OrganizationContext';

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;

    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`simple-tabpanel-${index}`}
            aria-labelledby={`simple-tab-${index}`}
            {...other}
        >
            {value === index && (
                <Box sx={{ p: 3 }}>
                    {children}
                </Box>
            )}
        </div>
    );
}

export const WorkerDetailView: React.FC = () => {
    const { orgId, workerId } = useParams<{ orgId: string; workerId: string }>();
    const { currentOrganization } = useOrganization();
    const [worker, setWorker] = useState<WorkerWithRuns | null>(null);
    const [logs, setLogs] = useState<string>('');
    const [loading, setLoading] = useState(false);
    const [logsLoading, setLogsLoading] = useState(false);
    const [actionLoading, setActionLoading] = useState(false);
    const [lastRefresh, setLastRefresh] = useState<Date>();
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [snackbarMessage, setSnackbarMessage] = useState<string>('');
    const [showSnackbar, setShowSnackbar] = useState(false);
    const [tabValue, setTabValue] = useState(0);
    const AUTO_REFRESH_INTERVAL = 10000; // 10 seconds

    const fetchWorker = useCallback(async () => {
        if (!orgId || !workerId) return;
        setLoading(true);
        try {
            const data = await api.getWorker(orgId, workerId);
            setWorker(data);
            setLastRefresh(new Date());
        } catch (error) {
            console.error('Error fetching worker:', error);
            setSnackbarMessage('Error fetching worker details');
            setShowSnackbar(true);
        } finally {
            setLoading(false);
        }
    }, [orgId, workerId]);

    const fetchLogs = useCallback(async () => {
        if (!orgId || !workerId) return;
        // setLogsLoading(true);
        try {
            const logs = await api.getWorkerLogs(orgId, workerId);
            setLogs(logs);
        } catch (error) {
            console.error('Error fetching worker logs:', error);
            setLogs('Error fetching logs: ' + (error instanceof Error ? error.message : String(error)));
        } finally {
            setLogsLoading(false);
        }
    }, [orgId, workerId]);

    const handleShutdown = async () => {
        if (!orgId || !workerId) return;
        setActionLoading(true);
        try {
            await api.shutdownWorker(orgId, workerId);
            await fetchWorker();
            setSnackbarMessage('Worker shutdown initiated');
            setShowSnackbar(true);
        } catch (error) {
            setSnackbarMessage('Error shutting down worker');
            setShowSnackbar(true);
        } finally {
            setActionLoading(false);
        }
    };

    useEffect(() => {
        fetchWorker();
        fetchLogs();
    }, [fetchWorker, fetchLogs]);

    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (autoRefresh && worker?.status === 'active') {
            interval = setInterval(() => {
                fetchWorker();
                if (tabValue === 1) { // Only fetch logs if logs tab is active
                    fetchLogs();
                }
            }, AUTO_REFRESH_INTERVAL);
        }
        return () => {
            if (interval) {
                clearInterval(interval);
            }
        };
    }, [autoRefresh, fetchWorker, fetchLogs, worker?.status, tabValue]);

    if (!orgId || !currentOrganization || !worker) {
        return <Typography>Loading...</Typography>;
    }

    const canShutdown = worker.status === 'active' || worker.status === 'starting';

    return (
        <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <Typography variant="h4" sx={{ flexGrow: 1 }}>Worker: {worker.id}</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    {canShutdown && (
                        <LoadingButton
                            loading={actionLoading}
                            variant="contained"
                            color="error"
                            onClick={handleShutdown}
                            startIcon={<PowerSettingsNewIcon />}
                        >
                            Shutdown Worker
                        </LoadingButton>
                    )}
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
                        onRefresh={() => {
                            fetchWorker();
                            if (tabValue === 1) {
                                fetchLogs();
                            }
                        }}
                        loading={loading || (tabValue === 1 && logsLoading)}
                        lastRefresh={lastRefresh}
                    />
                </Box>
            </Box>

            <Box sx={{ mb: 3 }}>
                <Chip label={`Status: ${worker.status}`} sx={{ mr: 1 }} />
                {worker.gpu_type && (
                    <Chip label={`GPU: ${worker.gpu_type} (${worker.vram_gb}GB)`} sx={{ mr: 1 }} />
                )}
                {worker.docker_image && (
                    <Chip label={`Image: ${worker.docker_image}`} sx={{ mr: 1 }} />
                )}
            </Box>

            <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
                <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
                    <Tab label="Details" />
                    <Tab label="Logs" />
                </Tabs>
            </Box>

            <TabPanel value={tabValue} index={0}>
                {worker.cached_models && worker.cached_models.length > 0 && (
                    <Box sx={{ mb: 3 }}>
                        <Typography variant="h6">Cached Models:</Typography>
                        <Paper sx={{ p: 2, bgcolor: 'grey.100' }}>
                            {worker.cached_models.map((model, index) => (
                                <Chip
                                    key={index}
                                    label={model}
                                    sx={{ m: 0.5 }}
                                />
                            ))}
                        </Paper>
                    </Box>
                )}

                <Typography variant="h6">Run History:</Typography>
                <List>
                    {worker.runs
                        .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
                        .map(run => (
                            <ListItem key={run.id} component={Link} to={`/${orgId}/runs/${run.id}`}>
                                <ListItemText
                                    primary={run.id}
                                    secondary={`Status: ${run.status}, Job: ${run.job_id}, Created: ${new Date(run.created_at).toLocaleString()}`}
                                />
                            </ListItem>
                        ))
                    }
                </List>
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
                <Paper
                    sx={{
                        p: 2,
                        bgcolor: 'grey.100',
                        maxHeight: '500px',
                        overflow: 'auto'
                    }}
                >
                    {logsLoading ? (
                        <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                            <CircularProgress />
                        </Box>
                    ) : (
                        <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
                            {logs || 'No logs available'}
                        </pre>
                    )}
                </Paper>
            </TabPanel>

            <Snackbar
                open={showSnackbar}
                autoHideDuration={6000}
                onClose={() => setShowSnackbar(false)}
                message={snackbarMessage}
            />
        </Paper>
    );
};
