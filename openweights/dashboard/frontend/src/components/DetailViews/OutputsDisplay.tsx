import React from 'react';
import { Typography, Paper, Box } from '@mui/material';
import { FileContent } from './FileContent';
import { MetricsDisplay } from './MetricsDisplay';

const isFileId = (key: string, value: any): boolean => {
    if (typeof value !== 'string') return false;
    return key.toLowerCase().includes('file') ||
           value.toString().startsWith('file-') ||
           value.toString().includes(':file-');
};

interface OutputsDisplayProps {
    outputs: any;
    orgId: string;
}

export const OutputsDisplay: React.FC<OutputsDisplayProps> = ({ outputs, orgId }) => {
    if (!outputs) return null;

    // Check if outputs is a metrics object (has numeric values) or contains file references
    const hasMetrics = Object.values(outputs).some(value => typeof value === 'number');

    if (hasMetrics) {
        return <MetricsDisplay metrics={outputs} />;
    }

    return (
        <Box>
            {Object.entries(outputs).map(([key, value]) => {
                if (isFileId(key, value)) {
                    return (
                        <Box key={key} sx={{ mb: 2 }}>
                            <Typography variant="subtitle1">{key}:</Typography>
                            <FileContent fileId={value as string} orgId={orgId} />
                        </Box>
                    );
                }

                if (typeof value === 'object') {
                    return (
                        <Box key={key} sx={{ mb: 2 }}>
                            <Typography variant="subtitle1">{key}:</Typography>
                            <Paper sx={{ p: 2, bgcolor: 'grey.100' }}>
                                <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
                                    {JSON.stringify(value, null, 2)}
                                </pre>
                            </Paper>
                        </Box>
                    );
                }

                return (
                    <Box key={key} sx={{ mb: 2 }}>
                        <Typography variant="subtitle1">{key}:</Typography>
                        <Typography>{String(value)}</Typography>
                    </Box>
                );
            })}
        </Box>
    );
};
