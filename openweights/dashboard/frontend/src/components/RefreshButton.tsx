import React from 'react';
import { IconButton, Tooltip, CircularProgress } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';

interface RefreshButtonProps {
    onRefresh: () => void;
    loading?: boolean;
    lastRefresh?: Date;
}

export const RefreshButton: React.FC<RefreshButtonProps> = ({ onRefresh, loading, lastRefresh }) => {
    const tooltipTitle = lastRefresh
        ? `Last refreshed: ${lastRefresh.toLocaleTimeString()}`
        : 'Refresh';

    return (
        <Tooltip title={tooltipTitle}>
            <span>  {/* Wrap in span to make Tooltip work with disabled button */}
                <IconButton
                    onClick={onRefresh}
                    disabled={loading}
                    size="small"
                >
                    {loading ? (
                        <CircularProgress size={20} />
                    ) : (
                        <RefreshIcon />
                    )}
                </IconButton>
            </span>
        </Tooltip>
    );
};
