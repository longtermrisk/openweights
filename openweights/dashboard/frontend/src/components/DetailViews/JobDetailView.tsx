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
    Snackbar
} from '@mui/material';
import { LoadingButton } from '@mui/lab';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import CancelIcon from '@mui/icons-material/Cancel';
import { JobWithRuns } from '../../types';
import { api } from '../../api';
import { RefreshButton } from '../RefreshButton';
import { useOrganization } from '../../contexts/OrganizationContext';
import { OutputsDisplay } from './OutputsDisplay';

export const JobDetailView: React.FC = () => {
    const { orgId, jobId } = useParams<{ orgId: string; jobId: string }>();
    const { currentOrganization } = useOrganization();
    const [job, setJob] = useState<JobWithRuns | null>(null);
    const [loading, setLoading] = useState(false);
    const [actionLoading, setActionLoading] = useState<'cancel' | 'restart' | null>(null);
    const [lastRefresh, setLastRefresh] = useState<Date>();
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [snackbarMessage, setSnackbarMessage] = useState<string>('');
    const [showSnackbar, setShowSnackbar] = useState(false);
    const AUTO_REFRESH_INTERVAL = 10000; // 10 seconds

    const fetchJob = useCallback(async () => {
        if (!orgId || !jobId) return;
        setLoading(true);
        try {
            const data = await api.getJob(orgId, jobId);
            setJob(data);
            setLastRefresh(new Date());
        } catch (error) {
            console.error('Error fetching job:', error);
            setSnackbarMessage('Error fetching job details');
            setShowSnackbar(true);
        } finally {
            setLoading(false);
        }
    }, [orgId, jobId]);

    const handleCancel = async () => {
        if (!orgId || !jobId) return;
        setActionLoading('cancel');
        try {
            await api.cancelJob(orgId, jobId);
            await fetchJob();
            setSnackbarMessage('Job cancelled successfully');
            setShowSnackbar(true);
        } catch (error) {
            console.error('Error cancelling job:', error);
            setSnackbarMessage('Error cancelling job');
            setShowSnackbar(true);
        } finally {
            setActionLoading(null);
        }
    };

    const handleRestart = async () => {
        if (!orgId || !jobId) return;
        setActionLoading('restart');
        try {
            await api.restartJob(orgId, jobId);
            await fetchJob();
            setSnackbarMessage('Job restarted successfully');
            setShowSnackbar(true);
        } catch (error) {
            console.error('Error restarting job:', error);
            setSnackbarMessage('Error restarting job');
            setShowSnackbar(true);
        } finally {
            setActionLoading(null);
        }
    };

    useEffect(() => {
        fetchJob();
    }, [fetchJob]);

    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (autoRefresh && job?.status === 'in_progress') {
            interval = setInterval(fetchJob, AUTO_REFRESH_INTERVAL);
        }
        return () => {
            if (interval) {
                clearInterval(interval);
            }
        };
    }, [autoRefresh, fetchJob, job?.status]);

    if (!orgId || !currentOrganization || !job) {
        return <Typography>Loading...</Typography>;
    }

    const canCancel = job.status === 'pending' || job.status === 'in_progress';
    const canRestart = job.status === 'failed' || job.status === 'canceled';

    return (
        <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <Typography variant="h4" sx={{ flexGrow: 1 }}>Job: {job.id}</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    {canCancel && (
                        <LoadingButton
                            loading={actionLoading === 'cancel'}
                            variant="contained"
                            color="error"
                            onClick={handleCancel}
                            startIcon={<CancelIcon />}
                        >
                            Cancel Job
                        </LoadingButton>
                    )}
                    {canRestart && (
                        <LoadingButton
                            loading={actionLoading === 'restart'}
                            variant="contained"
                            color="primary"
                            onClick={handleRestart}
                            startIcon={<RestartAltIcon />}
                        >
                            Restart Job
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
                        onRefresh={fetchJob}
                        loading={loading}
                        lastRefresh={lastRefresh}
                    />
                </Box>
            </Box>

            <Box sx={{ mb: 3 }}>
                <Chip label={`Status: ${job.status}`} sx={{ mr: 1 }} />
                <Chip label={`Type: ${job.type}`} sx={{ mr: 1 }} />
                {job.model && <Chip label={`Model: ${job.model}`} sx={{ mr: 1 }} />}
                {job.docker_image && <Chip label={`Image: ${job.docker_image}`} sx={{ mr: 1 }} />}
            </Box>

            {job.script && (
                <Box sx={{ mb: 3 }}>
                    <Typography variant="h6">Script:</Typography>
                    <Paper sx={{ p: 2, bgcolor: 'grey.100' }}>
                        <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
                            {job.script}
                        </pre>
                    </Paper>
                </Box>
            )}

            {job.params && (
                <Box sx={{ mb: 3 }}>
                    <Typography variant="h6">Parameters:</Typography>
                    <Paper sx={{ p: 2, bgcolor: 'grey.100' }}>
                        <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
                            {JSON.stringify(job.params, null, 2)}
                        </pre>
                    </Paper>
                </Box>
            )}

            {job.outputs && (
                <Box sx={{ mb: 3 }}>
                    <Typography variant="h6">Outputs:</Typography>
                    <OutputsDisplay outputs={job.outputs} orgId={orgId} />
                </Box>
            )}

            <Typography variant="h6">Runs:</Typography>
            <List>
                {job.runs
                    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
                    .map(run => (
                        <ListItem key={run.id} component={Link} to={`/${orgId}/runs/${run.id}`}>
                            <ListItemText
                                primary={run.id}
                                secondary={`Status: ${run.status}, Created: ${new Date(run.created_at).toLocaleString()}`}
                            />
                        </ListItem>
                    ))
                }
            </List>

            <Snackbar
                open={showSnackbar}
                autoHideDuration={6000}
                onClose={() => setShowSnackbar(false)}
                message={snackbarMessage}
            />
        </Paper>
    );
};
