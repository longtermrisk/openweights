import React, { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
    Grid,
    Paper,
    Typography,
    Card,
    CardContent,
    Button,
    Box,
    TextField,
    TablePagination,
    Chip,
    Alert,
    LinearProgress
} from '@mui/material';
import { Job } from '../types';
import { api } from '../api';
import { StatusCheckboxes, StatusFilters } from './StatusCheckboxes';
import { ViewToggle } from './ViewToggle';
import { JobsListView } from './JobsListView';
import { useOrganization } from '../contexts/OrganizationContext';

const JobCard: React.FC<{ job: Job; orgId: string; onCancelJob: (jobId: string) => Promise<void>; onRetryJob: (jobId: string) => Promise<void> }> = ({ job, orgId, onCancelJob, onRetryJob }) => (
    <Card
        sx={{
            mb: 1,
            backgroundColor: '#ffffff',
            transition: 'background-color 0.3s ease',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            position: 'relative'
        }}
    >
        <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
            {/* Status chip in top right corner */}
            <Box sx={{ position: 'absolute', top: 8, right: 8 }}>
                <Chip
                    label={job.status}
                    color={
                        job.status === 'completed' ? 'success' :
                        job.status === 'failed' ? 'error' :
                        job.status === 'canceled' ? 'warning' :
                        job.status === 'in_progress' ? 'info' :
                        'default'
                    }
                    size="small"
                />
            </Box>

            {/* Job ID as clickable link */}
            <Button
                component={Link}
                to={`/${orgId}/jobs/${job.id}`}
                sx={{
                    p: 0,
                    minWidth: 0,
                    textTransform: 'none',
                    justifyContent: 'flex-start',
                    '&:hover': { backgroundColor: 'transparent', textDecoration: 'underline' },
                    mb: 0.5,
                    pr: 10  // Make room for status chip
                }}
            >
                <Typography variant="h6" component="span" color="primary" sx={{ fontSize: '0.95rem' }}>
                    {job.id}
                </Typography>
            </Button>

            {/* Model info if available */}
            {job.model && (
                <Typography color="text.secondary" sx={{ fontSize: '0.85rem', mb: 0.5 }}>
                    Model: {job.model}
                </Typography>
            )}

            {/* Created date and Cancel/Retry button on same row */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography color="text.secondary" sx={{ fontSize: '0.8rem' }}>
                    {new Date(job.created_at).toLocaleString()}
                </Typography>
                {(job.status === 'pending' || job.status === 'in_progress') && (
                    <Button
                        variant="text"
                        color="error"
                        size="small"
                        onClick={() => onCancelJob(String(job.id))}
                        sx={{ minWidth: 0, p: 0.5, fontSize: '0.75rem' }}
                    >
                        Cancel
                    </Button>
                )}
                {(job.status === 'failed' || job.status === 'canceled') && (
                    <Button
                        variant="text"
                        color="primary"
                        size="small"
                        onClick={() => onRetryJob(String(job.id))}
                        sx={{ minWidth: 0, p: 0.5, fontSize: '0.75rem' }}
                    >
                        Retry
                    </Button>
                )}
            </Box>
        </CardContent>
    </Card>
);

interface JobsColumnProps {
    title: string;
    jobs: Job[];
    total: number;
    page: number;
    rowsPerPage: number;
    onPageChange: (newPage: number) => void;
    onRowsPerPageChange: (newRowsPerPage: number) => void;
    orgId: string;
    onCancelJob: (jobId: string) => Promise<void>;
    onRetryJob: (jobId: string) => Promise<void>;
}

const JobsColumn: React.FC<JobsColumnProps> = ({
    title,
    jobs,
    total,
    page,
    rowsPerPage,
    onPageChange,
    onRowsPerPageChange,
    orgId,
    onCancelJob,
    onRetryJob
}) => {

    return (
        <Grid item xs={12} md={4} sx={{ height: '100%' }}>
            <Paper
                sx={{
                    p: 1,
                    height: '100%',
                    overflow: 'auto',
                    display: 'flex',
                    flexDirection: 'column',
                    backgroundColor: '#ffffff',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}
            >
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Typography variant="h5" sx={{ color: 'text.primary' }}>
                        {title} ({total})
                    </Typography>
                </Box>
                <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 1 }}>
                    {jobs.map(job => (
                        <JobCard key={job.id} job={job} orgId={orgId} onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
                    ))}
                </Box>
                <TablePagination
                    component="div"
                    count={total}
                    page={page}
                    onPageChange={(_, newPage) => onPageChange(newPage)}
                    rowsPerPage={rowsPerPage}
                    onRowsPerPageChange={(event) => onRowsPerPageChange(parseInt(event.target.value, 10))}
                    rowsPerPageOptions={[5, 10, 25]}
                />
            </Paper>
        </Grid>
    );
};

export const JobsView: React.FC = () => {
    const { orgId } = useParams<{ orgId: string }>();
    const { currentOrganization } = useOrganization();
    const [filter, setFilter] = useState('');
    const [search, setSearch] = useState('');
    const [pages, setPages] = useState({ pending: 0, inProgress: 0, completed: 0 });
    const [rowsPerPage, setRowsPerPage] = useState(10);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [refreshVersion, setRefreshVersion] = useState(0);
    const [view, setView] = useState<'three-column' | 'list'>('three-column');
    const [statusFilters, setStatusFilters] = useState<StatusFilters>({
        completed: true, failed: true, canceled: true
    });
    const emptyPage = { items: [] as Job[], total: 0 };
    const [results, setResults] = useState({ pending: emptyPage, inProgress: emptyPage, completed: emptyPage });

    useEffect(() => {
        if (filter === search) return;
        const timer = setTimeout(() => {
            setSearch(filter);
            setPages({ pending: 0, inProgress: 0, completed: 0 });
        }, 300);
        return () => clearTimeout(timer);
    }, [filter, search]);

    useEffect(() => {
        if (!orgId) return;
        const controller = new AbortController();
        let timer: ReturnType<typeof setTimeout>;
        const finished = Object.entries(statusFilters).filter(([, enabled]) => enabled).map(([status]) => status);
        const empty = { items: [] as Job[], total: 0 };
        setResults({ pending: empty, inProgress: empty, completed: empty });
        const fetchPage = (statuses: string[], page: number) => statuses.length
            ? api.getJobsPage(orgId, statuses, search, rowsPerPage, page * rowsPerPage, controller.signal)
            : Promise.resolve(empty);
        const fetchJobs = async () => {
            setLoading(true);
            try {
                const requests = view === 'three-column'
                    ? [
                        fetchPage(['pending'], pages.pending),
                        fetchPage(['in_progress'], pages.inProgress),
                        fetchPage(finished, pages.completed)
                    ]
                    : [Promise.resolve(empty), Promise.resolve(empty), fetchPage(['pending', 'in_progress', ...finished], pages.completed)];
                // Wait for every column even when one fails, so polling cannot overlap
                // a slow request that is still running after a sibling's error.
                const responses = await Promise.allSettled(requests);
                const [pending, inProgress, completed] = responses.map(response => {
                    if (response.status === 'rejected') throw response.reason;
                    return response.value;
                });
                if (controller.signal.aborted) return;
                setResults({ pending, inProgress, completed });
                setError('');
                const clamp = (page: number, total: number) => Math.min(page, Math.max(0, Math.ceil(total / rowsPerPage) - 1));
                const next = {
                    pending: clamp(pages.pending, pending.total),
                    inProgress: clamp(pages.inProgress, inProgress.total),
                    completed: clamp(pages.completed, completed.total)
                };
                if (Object.keys(pages).some(key => pages[key as keyof typeof pages] !== next[key as keyof typeof next])) {
                    setPages(next);
                }
            } catch (error) {
                if (!controller.signal.aborted) setError(error instanceof Error ? error.message : 'Failed to load jobs');
            } finally {
                if (!controller.signal.aborted) {
                    setLoading(false);
                    // Schedule after completion so slow requests never pile up.
                    timer = setTimeout(fetchJobs, 10000);
                }
            }
        };
        fetchJobs();
        return () => { controller.abort(); clearTimeout(timer); };
    }, [orgId, search, pages, rowsPerPage, statusFilters, view, refreshVersion]);

    const handlePageChange = (status: string) => (newPage: number) => {
        setPages(prev => ({ ...prev, [status]: newPage }));
    };
    const resetPages = () => setPages({ pending: 0, inProgress: 0, completed: 0 });
    const handleRowsPerPageChange = (newRowsPerPage: number) => {
        setRowsPerPage(newRowsPerPage);
        resetPages();
    };
    const cancelJob = async (jobId: string) => {
        if (!orgId) return;
        try {
            await api.cancelJob(orgId, jobId);
            setRefreshVersion(version => version + 1);
        } catch (error) {
            setError(error instanceof Error ? error.message : 'Failed to cancel job');
        }
    };
    const retryJob = async (jobId: string) => {
        if (!orgId) return;
        try {
            await api.restartJob(orgId, jobId);
            setRefreshVersion(version => version + 1);
        } catch (error) {
            setError(error instanceof Error ? error.message : 'Failed to retry job');
        }
    };

    if (!orgId || !currentOrganization) {
        return <Typography>Loading...</Typography>;
    }

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box
                sx={{
                    mb: 1.5,
                    display: 'flex',
                    gap: 2,
                    alignItems: 'center',
                    flexWrap: 'wrap',
                    p: 1,
                    backgroundColor: '#ffffff',
                    borderRadius: 1,
                    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}
            >
                <TextField
                    label="Search"
                    variant="outlined"
                    size="small"
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                    sx={{
                        width: 200,
                        '& .MuiOutlinedInput-root': {
                            backgroundColor: '#ffffff',
                        }
                    }}
                />
                <StatusCheckboxes
                    filters={statusFilters}
                    onChange={filters => { setStatusFilters(filters); resetPages(); }}
                />
                <Box sx={{ ml: 'auto' }}>
                    <ViewToggle view={view} onViewChange={next => { setView(next); resetPages(); }} />
                </Box>
            </Box>
            {loading && <LinearProgress aria-label="Loading jobs" />}
            {error && <Alert severity="error">{error}</Alert>}
            {view === 'three-column' ? (
                <Grid container spacing={1.5} sx={{ flexGrow: 1 }}>
                    <JobsColumn
                        title="Pending"
                        jobs={results.pending.items}
                        total={results.pending.total}
                        page={pages.pending}
                        rowsPerPage={rowsPerPage}
                        onPageChange={handlePageChange('pending')}
                        onRowsPerPageChange={handleRowsPerPageChange}
                        orgId={orgId}
                        onCancelJob={cancelJob}
                        onRetryJob={retryJob}
                    />
                    <JobsColumn
                        title="In Progress"
                        jobs={results.inProgress.items}
                        total={results.inProgress.total}
                        page={pages.inProgress}
                        rowsPerPage={rowsPerPage}
                        onPageChange={handlePageChange('inProgress')}
                        onRowsPerPageChange={handleRowsPerPageChange}
                        orgId={orgId}
                        onCancelJob={cancelJob}
                        onRetryJob={retryJob}
                    />
                    <JobsColumn
                        title="Finished"
                        jobs={results.completed.items}
                        total={results.completed.total}
                        page={pages.completed}
                        rowsPerPage={rowsPerPage}
                        onPageChange={handlePageChange('completed')}
                        onRowsPerPageChange={handleRowsPerPageChange}
                        orgId={orgId}
                        onCancelJob={cancelJob}
                        onRetryJob={retryJob}
                    />
                </Grid>
            ) : (
                <JobsListView
                    jobs={results.completed.items}
                    total={results.completed.total}
                    page={pages.completed}
                    rowsPerPage={rowsPerPage}
                    onPageChange={(_, newPage) => handlePageChange('completed')(newPage)}
                    onRowsPerPageChange={(event) => handleRowsPerPageChange(parseInt(event.target.value, 10))}
                    orgId={orgId}
                    onCancelJob={cancelJob}
                    onRetryJob={retryJob}
                />
            )}
        </Box>
    );
};
