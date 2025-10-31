import React, { useEffect, useState, useCallback } from 'react';
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
    Chip
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
    filter: string;
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
    filter,
    page,
    rowsPerPage,
    onPageChange,
    onRowsPerPageChange,
    orgId,
    onCancelJob,
    onRetryJob
}) => {

    const filteredJobs = jobs.filter(job => {
        const searchStr = filter.toLowerCase();
        const jobId = String(job.id);
        const model = job.model ? job.model.toLowerCase() : '';
        const dockerImage = job.docker_image ? job.docker_image.toLowerCase() : '';

        return jobId.includes(searchStr) ||
            model.includes(searchStr) ||
            dockerImage.includes(searchStr) ||
            JSON.stringify(job.params).toLowerCase().includes(searchStr) ||
            JSON.stringify(job.outputs).toLowerCase().includes(searchStr);
    });

    const paginatedJobs = filteredJobs.slice(
        page * rowsPerPage,
        page * rowsPerPage + rowsPerPage
    );

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
                        {title} ({filteredJobs.length})
                    </Typography>
                </Box>
                <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 1 }}>
                    {paginatedJobs.map(job => (
                        <JobCard key={job.id} job={job} orgId={orgId} onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
                    ))}
                </Box>
                <TablePagination
                    component="div"
                    count={filteredJobs.length}
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
    const [jobs, setJobs] = useState<Job[]>([]);
    const [filter, setFilter] = useState('');
    const [typeFilter] = useState('all');
    const [pages, setPages] = useState({ pending: 0, inProgress: 0, completed: 0 });
    const [rowsPerPage, setRowsPerPage] = useState(10);
    const [, setLoading] = useState(false);
    // const [isCancelling, setIsCancelling] = useState<string | null>(null);
    const [view, setView] = useState<'three-column' | 'list'>('three-column');
    const [statusFilters, setStatusFilters] = useState<StatusFilters>({
        completed: true,
        failed: true,
        canceled: true
    });
    const AUTO_REFRESH_INTERVAL = 10000; // 10 seconds

    const fetchJobs = useCallback(async () => {
        if (!orgId) return;

        setLoading(true);
        try {
            const data = await api.getJobs(orgId);
            // Sort by created_at descending
            data.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
            setJobs(data);
        } catch (error) {
            console.error('Error fetching jobs:', error);
        } finally {
            setLoading(false);
        }
    }, [orgId]);

    useEffect(() => {
        fetchJobs();
    }, [fetchJobs]);

    useEffect(() => {
        const interval = setInterval(fetchJobs, AUTO_REFRESH_INTERVAL);
        return () => clearInterval(interval);
    }, [fetchJobs]);

    const handlePageChange = (status: string) => (newPage: number) => {
        setPages(prev => ({ ...prev, [status]: newPage }));
    };

    const handleRowsPerPageChange = (newRowsPerPage: number) => {
        setRowsPerPage(newRowsPerPage);
        setPages({ pending: 0, inProgress: 0, completed: 0 });
    };

    const cancelJob = async (jobId: string) => {
        if (!orgId) return;
        try {
            await api.cancelJob(orgId, jobId);
            await fetchJobs();
        } catch (error) {
            console.error('Failed to cancel job', error);
        }
    };

    const retryJob = async (jobId: string) => {
        if (!orgId) return;
        try {
            await api.restartJob(orgId, jobId);
            await fetchJobs();
        } catch (error) {
            console.error('Failed to retry job', error);
        }
    };

    const filteredJobs = jobs.filter(job => {
        const matchesType = typeFilter === 'all' || job.type === typeFilter;
        return matchesType;
    });

    const pendingJobs = filteredJobs.filter(job => job.status === 'pending');
    const inProgressJobs = filteredJobs.filter(job => job.status === 'in_progress');
    const finishedJobs = filteredJobs.filter(job => {
        if (job.status === 'completed' && statusFilters.completed) return true;
        if (job.status === 'failed' && statusFilters.failed) return true;
        if (job.status === 'canceled' && statusFilters.canceled) return true;
        return false;
    });

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
                    onChange={setStatusFilters}
                />
                <Box sx={{ ml: 'auto' }}>
                    <ViewToggle view={view} onViewChange={setView} />
                </Box>
            </Box>
            {view === 'three-column' ? (
                <Grid container spacing={1.5} sx={{ flexGrow: 1 }}>
                    <JobsColumn
                        title="Pending"
                        jobs={pendingJobs}
                        filter={filter}
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
                        jobs={inProgressJobs}
                        filter={filter}
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
                        jobs={finishedJobs}
                        filter={filter}
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
                    jobs={[...pendingJobs, ...inProgressJobs, ...finishedJobs]}
                    filter={filter}
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
