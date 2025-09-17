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
import { Run } from '../types';

const getStatusChipColor = (status: string) => {
    switch (status) {
        case 'completed':
            return 'success';
        case 'failed':
            return 'error';
        case 'canceled':
            return 'warning';
        case 'in_progress':
            return 'info';
        default:
            return 'default';
    }
};

interface RunsListViewProps {
    runs: Run[];
    filter: string;
    page: number;
    rowsPerPage: number;
    onPageChange: (event: unknown, newPage: number) => void;
    onRowsPerPageChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
    orgId: string;
}

export const RunsListView: React.FC<RunsListViewProps> = ({
    runs,
    filter,
    page,
    rowsPerPage,
    onPageChange,
    onRowsPerPageChange,
    orgId,
}) => {
    const filteredRuns = runs.filter(run => {
        const searchStr = filter.toLowerCase();
        const runId = String(run.id);
        const jobId = String(run.job_id);
        const workerId = run.worker_id ? String(run.worker_id) : '';

        return runId.toLowerCase().includes(searchStr) ||
            jobId.toLowerCase().includes(searchStr) ||
            workerId.toLowerCase().includes(searchStr);
    });

    const emptyRows = page > 0 ? Math.max(0, (1 + page) * rowsPerPage - filteredRuns.length) : 0;

    return (
        <Box sx={{ width: '100%' }}>
            <TableContainer component={Paper}>
                <Table sx={{ minWidth: 650 }} aria-label="runs table">
                    <TableHead>
                        <TableRow>
                            <TableCell>ID</TableCell>
                            <TableCell>Job ID</TableCell>
                            <TableCell>Worker ID</TableCell>
                            <TableCell>Status</TableCell>
                            <TableCell>Created At</TableCell>
                            <TableCell>Actions</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {filteredRuns
                            .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                            .map((run) => (
                                <TableRow key={run.id}>
                                    <TableCell component="th" scope="row">
                                        {run.id}
                                    </TableCell>
                                    <TableCell>
                                        <Link to={`/${orgId}/jobs/${run.job_id}`}>{run.job_id}</Link>
                                    </TableCell>
                                    <TableCell>
                                        {run.worker_id ? (
                                            <Link to={`/${orgId}/workers/${run.worker_id}`}>{run.worker_id}</Link>
                                        ) : '-'}
                                    </TableCell>
                                    <TableCell>
                                        <Chip
                                            label={run.status}
                                            color={getStatusChipColor(run.status) as any}
                                            size="small"
                                        />
                                    </TableCell>
                                    <TableCell>{new Date(run.created_at).toLocaleString()}</TableCell>
                                    <TableCell>
                                        <Button
                                            component={Link}
                                            to={`/${orgId}/runs/${run.id}`}
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
                                <TableCell colSpan={6} />
                            </TableRow>
                        )}
                    </TableBody>
                </Table>
            </TableContainer>
            <TablePagination
                rowsPerPageOptions={[5, 10, 25]}
                component="div"
                count={filteredRuns.length}
                rowsPerPage={rowsPerPage}
                page={page}
                onPageChange={onPageChange}
                onRowsPerPageChange={onRowsPerPageChange}
            />
        </Box>
    );
};
