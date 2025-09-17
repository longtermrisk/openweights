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
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    TablePagination,
    Chip,
    FormControlLabel,
    Switch
} from '@mui/material';
import { Worker } from '../types';
import { api } from '../api';
import { RefreshButton } from './RefreshButton';
import { ViewToggle } from './ViewToggle';
import { WorkersListView } from './WorkersListView';
import { useOrganization } from '../contexts/OrganizationContext';

const WorkerCard: React.FC<{ worker: Worker; orgId: string }> = ({ worker, orgId }) => (
    <Card
        sx={{
            mb: 2,
            backgroundColor: '#ffffff',
            transition: 'background-color 0.3s ease',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }}
    >
        <CardContent>
            <Typography variant="h6" component="div">
                {worker.id}
            </Typography>
            <Box sx={{ mt: 1, mb: 1 }}>
                <Chip
                    label={worker.status}
                    color={
                        worker.status === 'active' ? 'success' :
                        worker.status === 'starting' ? 'warning' :
                        worker.status === 'shutdown' || worker.status === 'terminated' ? 'error' :
                        'default'
                    }
                    size="small"
                />
            </Box>
            {worker.gpu_type && (
                <Typography color="text.secondary">
                    GPU: {worker.gpu_count} x {worker.gpu_type} ({worker.vram_gb}GB)
                </Typography>
            )}
            {worker.docker_image && (
                <Typography color="text.secondary" sx={{
                    wordBreak: 'break-word',
                    mb: 1
                }}>
                    Image: {worker.docker_image}
                </Typography>
            )}
            <Typography color="text.secondary" sx={{ mb: 1 }}>
                Created: {new Date(worker.created_at).toLocaleString()}
            </Typography>
            {worker.ping && (
                <Typography color="text.secondary" sx={{ mb: 1 }}>
                    Last ping: {new Date(worker.ping).toLocaleString()}
                </Typography>
            )}
            {worker.cached_models && worker.cached_models.length > 0 && (
                <Box sx={{ mb: 1 }}>
                    <Typography color="text.secondary" sx={{ mb: 0.5 }}>
                        Cached Models:
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {worker.cached_models.map((model, index) => (
                            <Chip
                                key={index}
                                label={model}
                                size="small"
                                sx={{
                                    backgroundColor: 'rgba(25, 118, 210, 0.08)',
                                    color: 'text.primary'
                                }}
                            />
                        ))}
                    </Box>
                </Box>
            )}
            <Button
                component={Link}
                to={`/${orgId}/workers/${worker.id}`}
                variant="outlined"
                sx={{ mt: 1 }}
            >
                View Details
            </Button>
        </CardContent>
    </Card>
);


interface WorkersColumnProps {
    title: string;
    workers: Worker[];
    filter: string;
    page: number;
    rowsPerPage: number;
    onPageChange: (newPage: number) => void;
    onRowsPerPageChange: (newRowsPerPage: number) => void;
    lastRefresh?: Date;
    onRefresh: () => void;
    loading?: boolean;
    orgId: string;
}

const WorkersColumn: React.FC<WorkersColumnProps> = ({
    title,
    workers,
    filter,
    page,
    rowsPerPage,
    onPageChange,
    onRowsPerPageChange,
    lastRefresh,
    onRefresh,
    loading,
    orgId
}) => {
    const filteredWorkers = workers.filter(worker => {
        const searchStr = filter.toLowerCase();
        const workerId = String(worker.id);
        const gpuType = worker.gpu_type ? worker.gpu_type.toLowerCase() : '';
        const dockerImage = worker.docker_image ? worker.docker_image.toLowerCase() : '';
        const cachedModels = worker.cached_models ? worker.cached_models.join(' ').toLowerCase() : '';

        return workerId.includes(searchStr) ||
            gpuType.includes(searchStr) ||
            dockerImage.includes(searchStr) ||
            cachedModels.includes(searchStr);
    });

    const paginatedWorkers = filteredWorkers.slice(
        page * rowsPerPage,
        page * rowsPerPage + rowsPerPage
    );

    return (
        <Grid item xs={12} md={4} sx={{ height: '100%' }}>
            <Paper sx={{ p: 2, height: '100%', overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h5" sx={{ flexGrow: 1 }}>
                        {title} ({filteredWorkers.length})
                    </Typography>
                    <RefreshButton
                        onRefresh={onRefresh}
                        loading={loading}
                        lastRefresh={lastRefresh}
                    />
                </Box>
                <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 2 }}>
                    {paginatedWorkers.map(worker => (
                        <WorkerCard key={worker.id} worker={worker} orgId={orgId} />
                    ))}
                </Box>
                <TablePagination
                    component="div"
                    count={filteredWorkers.length}
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

export const WorkersView: React.FC = () => {
    const { orgId } = useParams<{ orgId: string }>();
    const { currentOrganization } = useOrganization();
    const [workers, setWorkers] = useState<Worker[]>([]);
    const [filter, setFilter] = useState('');
    const [gpuFilter, setGpuFilter] = useState('all');
    const [pages, setPages] = useState({ starting: 0, active: 0, terminated: 0 });
    const [rowsPerPage, setRowsPerPage] = useState(10);
    const [loading, setLoading] = useState(false);
    const [lastRefresh, setLastRefresh] = useState<Date>();
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [view, setView] = useState<'three-column' | 'list'>('three-column');
    const AUTO_REFRESH_INTERVAL = 10000; // 10 seconds

    const fetchWorkers = useCallback(async () => {
        if (!orgId) return;

        setLoading(true);
        try {
            const data = await api.getWorkers(orgId);
            // Sort by created_at descending
            data.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
            setWorkers(data);
            setLastRefresh(new Date());
        } catch (error) {
            console.error('Error fetching workers:', error);
        } finally {
            setLoading(false);
        }
    }, [orgId]);

    useEffect(() => {
        fetchWorkers();
    }, [fetchWorkers]);

    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (autoRefresh) {
            interval = setInterval(fetchWorkers, AUTO_REFRESH_INTERVAL);
        }
        return () => {
            if (interval) {
                clearInterval(interval);
            }
        };
    }, [autoRefresh, fetchWorkers]);

    const handlePageChange = (status: string) => (newPage: number) => {
        setPages(prev => ({ ...prev, [status]: newPage }));
    };

    const handleRowsPerPageChange = (newRowsPerPage: number) => {
        setRowsPerPage(newRowsPerPage);
        setPages({ starting: 0, active: 0, terminated: 0 });
    };

    // Get unique GPU types for filter
    const gpuTypes = Array.from(new Set(workers
        .map(w => w.gpu_type)
        .filter(Boolean) as string[]
    ));

    const filteredWorkers = workers.filter(worker => {
        const matchesGpu = gpuFilter === 'all' || worker.gpu_type === gpuFilter;
        return matchesGpu;
    });

    const startingWorkers = filteredWorkers.filter(worker => worker.status === 'starting');
    const activeWorkers = filteredWorkers.filter(worker => worker.status === 'active');
    const terminatedWorkers = filteredWorkers.filter(worker => worker.status === 'terminated' || worker.status === 'shutdown');

    if (!orgId || !currentOrganization) {
        return <Typography>Loading...</Typography>;
    }

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ mb: 3, display: 'flex', gap: 2, alignItems: 'center', p: 2, backgroundColor: '#ffffff', borderRadius: 1, boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
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
                <FormControl size="small" sx={{ minWidth: 120 }}>
                    <InputLabel>GPU Type</InputLabel>
                    <Select
                        value={gpuFilter}
                        label="GPU Type"
                        onChange={(e) => setGpuFilter(e.target.value)}
                    >
                        <MenuItem value="all">All</MenuItem>
                        {gpuTypes.map(type => (
                            <MenuItem key={type} value={type}>{type}</MenuItem>
                        ))}
                    </Select>
                </FormControl>
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
                <Box sx={{ ml: 'auto' }}>
                    <ViewToggle view={view} onViewChange={setView} />
                </Box>
            </Box>
            {view === 'three-column' ? (
                <Grid container spacing={3} sx={{ flexGrow: 1 }}>
                    <WorkersColumn
                        title="Starting"
                        workers={startingWorkers}
                        filter={filter}
                        page={pages.starting}
                        rowsPerPage={rowsPerPage}
                        onPageChange={handlePageChange('starting')}
                        onRowsPerPageChange={handleRowsPerPageChange}
                        lastRefresh={lastRefresh}
                        onRefresh={fetchWorkers}
                        loading={loading}
                        orgId={orgId}
                    />
                    <WorkersColumn
                        title="Active"
                        workers={activeWorkers}
                        filter={filter}
                        page={pages.active}
                        rowsPerPage={rowsPerPage}
                        onPageChange={handlePageChange('active')}
                        onRowsPerPageChange={handleRowsPerPageChange}
                        lastRefresh={lastRefresh}
                        onRefresh={fetchWorkers}
                        loading={loading}
                        orgId={orgId}
                    />
                    <WorkersColumn
                        title="Terminated/Shutdown"
                        workers={terminatedWorkers}
                        filter={filter}
                        page={pages.terminated}
                        rowsPerPage={rowsPerPage}
                        onPageChange={handlePageChange('terminated')}
                        onRowsPerPageChange={handleRowsPerPageChange}
                        lastRefresh={lastRefresh}
                        onRefresh={fetchWorkers}
                        loading={loading}
                        orgId={orgId}
                    />
                </Grid>
            ) : (
                <WorkersListView
                    workers={[...startingWorkers, ...activeWorkers, ...terminatedWorkers]}
                    filter={filter}
                    page={pages.terminated}
                    rowsPerPage={rowsPerPage}
                    onPageChange={(_, newPage) => handlePageChange('terminated')(newPage)}
                    onRowsPerPageChange={(event) => handleRowsPerPageChange(parseInt(event.target.value, 10))}
                    orgId={orgId}
                />
            )}
        </Box>
    );
};
