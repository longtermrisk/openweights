import React from 'react';
import { FormGroup, FormControlLabel, Checkbox, Box, Typography } from '@mui/material';

export interface StatusFilters {
    completed: boolean;
    failed: boolean;
    canceled: boolean;
}

interface StatusCheckboxesProps {
    filters: StatusFilters;
    onChange: (newFilters: StatusFilters) => void;
}

export const StatusCheckboxes: React.FC<StatusCheckboxesProps> = ({ filters, onChange }) => {
    const handleChange = (status: keyof StatusFilters) => (event: React.ChangeEvent<HTMLInputElement>) => {
        onChange({
            ...filters,
            [status]: event.target.checked
        });
    };

    return (
        <Box>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                Status Filters
            </Typography>
            <FormGroup row>
                <FormControlLabel
                    control={
                        <Checkbox
                            checked={filters.completed}
                            onChange={handleChange('completed')}
                            size="small"
                        />
                    }
                    label="Completed"
                />
                <FormControlLabel
                    control={
                        <Checkbox
                            checked={filters.failed}
                            onChange={handleChange('failed')}
                            size="small"
                        />
                    }
                    label="Failed"
                />
                <FormControlLabel
                    control={
                        <Checkbox
                            checked={filters.canceled}
                            onChange={handleChange('canceled')}
                            size="small"
                        />
                    }
                    label="Canceled"
                />
            </FormGroup>
        </Box>
    );
};
