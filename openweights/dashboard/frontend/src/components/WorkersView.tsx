import React, { useEffect, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
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

const getStatusColor = (status: string) => {
    switch (status) {
        case 'active':
            return '#e6f4ea';  // light green
        case 'starting':
            return '#fff8e1';  // light yellow
        case 'terminated':
            return '#ffebee';  // light red
        default:
            return undefined;
    }
};

const WorkerCard: React.FC<{ worker: Worker }> = ({ worker }) => (
    <Card sx={{ mb: 2, backgroundColor: getStatusColor(worker.status) }}>
        <CardContent>
            <Typography variant="h6" component="div">
                {worker.id}
            </Typography>
            <Typography color="text.secondary">
                Status: {worker.status}
            </Typography>
            {worker.gpu_type && (
                <Typography color="text.secondary">
                    GPU: {worker.gpu_type} ({worker.vram_gb}GB)
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
            <Button component={Link} to={`/workers/${worker.id}`} variant="outlined" sx={{ mt: 1 }}>
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
    loading
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
                        <WorkerCard key={worker.id} worker={worker} />
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
    const [workers, setWorkers] = useState<Worker[]>([]);
    const [filter, setFilter] = useState('');
    const [gpuFilter, setGpuFilter] = useState('all');
    const [pages, setPages] = useState({ starting: 0, active: 0, terminated: 0 });
    const [rowsPerPage, setRowsPerPage] = useState(10);
    const [loading, setLoading] = useState(false);
    const [lastRefresh, setLastRefresh] = useState<Date>();
    const [autoRefresh, setAutoRefresh] = useState(true);
    const AUTO_REFRESH_INTERVAL = 10000; // 10 seconds

    const fetchWorkers = useCallback(async () => {
        setLoading(true);
        try {
            const data = await api.getWorkers();
            // Sort by created_at descending
            data.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
            setWorkers(data);
            setLastRefresh(new Date());
        } catch (error) {
            console.error('Error fetching workers:', error);
        } finally {
            setLoading(false);
        }
    }, []);

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
    const terminatedWorkers = filteredWorkers.filter(worker => worker.status === 'terminated');

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ mb: 3, display: 'flex', gap: 2, alignItems: 'center' }}>
                <TextField
                    label="Search"
                    variant="outlined"
                    size="small"
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                    sx={{ 
                        width: 200,
                        '& .MuiOutlinedInput-root': {
                            backgroundColor: 'background.paper',
                            '& fieldset': {
                                borderColor: 'rgba(255, 255, 255, 0.23)',
                            },
                            '&:hover fieldset': {
                                borderColor: 'rgba(255, 255, 255, 0.4)',
                            },
                            '&.Mui-focused fieldset': {
                                borderColor: 'primary.main',
                            },
                        },
                        '& .MuiInputLabel-root': {
                            color: 'text.secondary',
                        },
                        '& .MuiInputBase-input': {
                            color: 'text.primary',
                        },
                    }}
                />
                <FormControl size="small" sx={{ 
                    minWidth: 120,
                    '& .MuiOutlinedInput-root': {
                        backgroundColor: 'background.paper',
                        '& fieldset': {
                            borderColor: 'rgba(255, 255, 255, 0.23)',
                        },
                        '&:hover fieldset': {
                            borderColor: 'rgba(255, 255, 255, 0.4)',
                        },
                        '&.Mui-focused fieldset': {
                            borderColor: 'primary.main',
                        },
                    },
                    '& .MuiInputLabel-root': {
                        color: 'text.secondary',
                    },
                    '& .MuiSelect-select': {
                        color: 'text.primary',
                    },
                }}>
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
            </Box>
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
                />
                <WorkersColumn 
                    title="Terminated" 
                    workers={terminatedWorkers}
                    filter={filter}
                    page={pages.terminated}
                    rowsPerPage={rowsPerPage}
                    onPageChange={handlePageChange('terminated')}
                    onRowsPerPageChange={handleRowsPerPageChange}
                    lastRefresh={lastRefresh}
                    onRefresh={fetchWorkers}
                    loading={loading}
                />
            </Grid>
        </Box>
    );
};