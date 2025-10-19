import React from 'react';
import { IconButton, Tooltip } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';

interface RefreshButtonProps {
    onRefresh: () => void;
    loading?: boolean;
    lastRefresh?: Date;
}

export const RefreshButton: React.FC<RefreshButtonProps> = ({ onRefresh, lastRefresh }) => {
    const tooltipTitle = lastRefresh
        ? `Last refreshed: ${lastRefresh.toLocaleTimeString()}`
        : 'Refresh';

    return (
        <Tooltip title={tooltipTitle}>
            <IconButton
                onClick={onRefresh}
                size="small"
            >
                <RefreshIcon />
            </IconButton>
        </Tooltip>
    );
};
