import React, { useEffect, useState } from 'react';
import { Typography, CircularProgress, Paper } from '@mui/material';
import { api } from '../../api';

interface FileContentProps {
    fileId: string;
    orgId: string;
}

export const FileContent: React.FC<FileContentProps> = ({ fileId, orgId }) => {
    const [content, setContent] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchContent = async () => {
            try {
                const data = await api.getFileContent(orgId, fileId);
                setContent(data);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load file content');
            } finally {
                setLoading(false);
            }
        };
        fetchContent();
    }, [fileId, orgId]);

    if (loading) return <CircularProgress size={20} />;
    if (error) return <Typography color="error">{error}</Typography>;
    if (!content) return <Typography>No content available</Typography>;

    return (
        <Paper sx={{ p: 2, bgcolor: 'grey.100', mt: 1 }}>
            <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
                {content}
            </pre>
        </Paper>
    );
};
