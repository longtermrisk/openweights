import React from 'react';
import { Typography, Paper, Box } from '@mui/material';

const formatMetricValue = (value: any): string => {
    if (typeof value === 'number') {
        // Format small numbers in scientific notation
        if (Math.abs(value) < 0.0001 && value !== 0) {
            return value.toExponential(6);
        }
        // Format regular numbers with up to 6 decimal places
        return value.toFixed(6).replace(/\.?0+$/, '');
    }
    return String(value);
};

interface MetricsDisplayProps {
    metrics: Record<string, any>;
}

export const MetricsDisplay: React.FC<MetricsDisplayProps> = ({ metrics }) => {
    // Group metrics by category
    const groups: Record<string, string[]> = {
        'Training': ['loss', 'nll_loss', 'eval_loss', 'eval_nll_loss', 'grad_norm', 'learning_rate'],
        'Progress': ['step', 'epoch'],
        'GPU': ['gpu_name', 'gpu_memory_free_mb', 'gpu_memory_reserved_mb', 'gpu_memory_allocated_mb'],
        'Model Outputs': [
            'logps/chosen', 'logps/rejected',
            'logits/chosen', 'logits/rejected',
            'log_odds_ratio', 'log_odds_chosen'
        ],
        'Rewards': [
            'rewards/chosen', 'rewards/rejected',
            'rewards/margins', 'rewards/accuracies'
        ],
        'Evaluation': [
            'eval_runtime', 'eval_steps_per_second',
            'eval_samples_per_second'
        ]
    };

    // Collect metrics by group
    const groupedMetrics: Record<string, Record<string, any>> = {};
    const ungroupedMetrics: Record<string, any> = {};

    Object.entries(metrics).forEach(([key, value]) => {
        let grouped = false;
        for (const [group, keys] of Object.entries(groups)) {
            if (keys.includes(key) || keys.some(k => key.startsWith(k))) {
                groupedMetrics[group] = groupedMetrics[group] || {};
                groupedMetrics[group][key] = value;
                grouped = true;
                break;
            }
        }
        if (!grouped) {
            ungroupedMetrics[key] = value;
        }
    });

    return (
        <Box>
            {Object.entries(groupedMetrics).map(([group, metrics]) => (
                metrics && Object.keys(metrics).length > 0 && (
                    <Box key={group} sx={{ mb: 3 }}>
                        <Typography variant="h6">{group}:</Typography>
                        <Paper sx={{ p: 2, bgcolor: 'grey.100' }}>
                            {Object.entries(metrics).map(([key, value]) => (
                                <Typography key={key} sx={{ fontFamily: 'monospace' }}>
                                    {key}: {formatMetricValue(value)}
                                </Typography>
                            ))}
                        </Paper>
                    </Box>
                )
            ))}
            {Object.keys(ungroupedMetrics).length > 0 && (
                <Box sx={{ mb: 3 }}>
                    <Typography variant="h6">Other:</Typography>
                    <Paper sx={{ p: 2, bgcolor: 'grey.100' }}>
                        {Object.entries(ungroupedMetrics).map(([key, value]) => (
                            <Typography key={key} sx={{ fontFamily: 'monospace' }}>
                                {key}: {formatMetricValue(value)}
                            </Typography>
                        ))}
                    </Paper>
                </Box>
            )}
        </Box>
    );
};
