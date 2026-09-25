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
import { Job } from '../types';

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

interface JobsListViewProps {
    jobs: Job[];
    total: number;
    page: number;
    rowsPerPage: number;
    onPageChange: (event: unknown, newPage: number) => void;
    onRowsPerPageChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
    orgId: string;
    onCancelJob: (jobId: string) => Promise<void>;
    onRetryJob: (jobId: string) => Promise<void>;
}

export const JobsListView: React.FC<JobsListViewProps> = ({
    jobs,
    total,
    page,
    rowsPerPage,
    onPageChange,
    onRowsPerPageChange,
    orgId,
    onCancelJob,
    onRetryJob,
}) => {
    return (
        <Box sx={{ width: '100%' }}>
            <TableContainer component={Paper}>
                <Table sx={{ minWidth: 650 }} aria-label="jobs table">
                    <TableHead>
                        <TableRow>
                            <TableCell>ID</TableCell>
                            <TableCell>Type</TableCell>
                            <TableCell>Status</TableCell>
                            <TableCell>Model</TableCell>
                            <TableCell>Docker Image</TableCell>
                            <TableCell>Created At</TableCell>
                            <TableCell>Actions</TableCell>
                            <TableCell>Manage</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {jobs
                            .map((job) => (
                                <TableRow key={job.id}>
                                    <TableCell component="th" scope="row">
                                        {job.id}
                                    </TableCell>
                                    <TableCell>{job.type}</TableCell>
                                    <TableCell>
                                        <Chip
                                            label={job.status}
                                            color={getStatusChipColor(job.status) as any}
                                            size="small"
                                        />
                                    </TableCell>
                                    <TableCell>{job.model || '-'}</TableCell>
                                    <TableCell sx={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                        {job.docker_image || '-'}
                                    </TableCell>
                                    <TableCell>{new Date(job.created_at).toLocaleString()}</TableCell>
                                    <TableCell>
                                        <Button
                                            component={Link}
                                            to={`/${orgId}/jobs/${job.id}`}
                                            size="small"
                                            variant="outlined"
                                        >
                                            View Details
                                        </Button>
                                    </TableCell>
                                    <TableCell>
                                        {(job.status === 'pending' || job.status === 'in_progress') && (
                                            <Button
                                                size="small"
                                                color="error"
                                                variant="outlined"
                                                onClick={() => onCancelJob(String(job.id))}
                                            >
                                                Cancel
                                            </Button>
                                        )}
                                        {(job.status === 'failed' || job.status === 'canceled') && (
                                            <Button
                                                size="small"
                                                color="primary"
                                                variant="outlined"
                                                onClick={() => onRetryJob(String(job.id))}
                                            >
                                                Retry
                                            </Button>
                                        )}
                                    </TableCell>
                                </TableRow>
                            ))}
                    </TableBody>
                </Table>
            </TableContainer>
            <TablePagination
                rowsPerPageOptions={[5, 10, 25]}
                component="div"
                count={total}
                rowsPerPage={rowsPerPage}
                page={page}
                onPageChange={onPageChange}
                onRowsPerPageChange={onRowsPerPageChange}
            />
        </Box>
    );
};
