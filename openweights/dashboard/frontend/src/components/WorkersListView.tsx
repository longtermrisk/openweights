import React from 'react';
import { Link } from 'react-router-dom';
import {
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    TablePagination,
    Button,
    Chip,
    Box
} from '@mui/material';
import { Worker } from '../types';

const getStatusChipColor = (status: string) => {
    switch (status) {
        case 'active':
            return 'success';
        case 'starting':
            return 'warning';
        case 'shutdown':
        case 'terminated':
            return 'error';
        default:
            return 'default';
    }
};

interface WorkersListViewProps {
    workers: Worker[];
    filter: string;
    page: number;
    rowsPerPage: number;
    onPageChange: (event: unknown, newPage: number) => void;
    onRowsPerPageChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
    orgId: string;
}

export const WorkersListView: React.FC<WorkersListViewProps> = ({
    workers,
    filter,
    page,
    rowsPerPage,
    onPageChange,
    onRowsPerPageChange,
    orgId,
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

    const emptyRows = page > 0 ? Math.max(0, (1 + page) * rowsPerPage - filteredWorkers.length) : 0;

    return (
        <Box sx={{ width: '100%' }}>
            <TableContainer component={Paper}>
                <Table sx={{ minWidth: 650 }} aria-label="workers table">
                    <TableHead>
                        <TableRow>
                            <TableCell>ID</TableCell>
                            <TableCell>Status</TableCell>
                            <TableCell>GPU</TableCell>
                            <TableCell>Docker Image</TableCell>
                            <TableCell>Last Ping</TableCell>
                            <TableCell>Created At</TableCell>
                            <TableCell>Actions</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {filteredWorkers
                            .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                            .map((worker) => (
                                <TableRow key={worker.id}>
                                    <TableCell component="th" scope="row">
                                        {worker.id}
                                    </TableCell>
                                    <TableCell>
                                        <Chip
                                            label={worker.status}
                                            color={getStatusChipColor(worker.status) as any}
                                            size="small"
                                        />
                                    </TableCell>
                                    <TableCell>
                                        {worker.gpu_type ?
                                            `${worker.gpu_count} x ${worker.gpu_type} (${worker.vram_gb}GB)` :
                                            '-'
                                        }
                                    </TableCell>
                                    <TableCell sx={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                        {worker.docker_image || '-'}
                                    </TableCell>
                                    <TableCell>
                                        {worker.ping ? new Date(worker.ping).toLocaleString() : '-'}
                                    </TableCell>
                                    <TableCell>{new Date(worker.created_at).toLocaleString()}</TableCell>
                                    <TableCell>
                                        <Button
                                            component={Link}
                                            to={`/${orgId}/workers/${worker.id}`}
                                            size="small"
                                            variant="outlined"
                                        >
                                            View Details
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                        {emptyRows > 0 && (
                            <TableRow style={{ height: 53 * emptyRows }}>
                                <TableCell colSpan={7} />
                            </TableRow>
                        )}
                    </TableBody>
                </Table>
            </TableContainer>
            <TablePagination
                rowsPerPageOptions={[5, 10, 25]}
                component="div"
                count={filteredWorkers.length}
                rowsPerPage={rowsPerPage}
                page={page}
                onPageChange={onPageChange}
                onRowsPerPageChange={onRowsPerPageChange}
            />
        </Box>
    );
};
