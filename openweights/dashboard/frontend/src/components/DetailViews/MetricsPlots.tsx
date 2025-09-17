import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { Box, Typography, CircularProgress } from '@mui/material';
import { api } from '../../api';

interface Event {
    data: Record<string, any>;
}

interface MetricsPlotsProps {
    orgId: string;
    runId: string;
}

export const MetricsPlots: React.FC<MetricsPlotsProps> = ({ orgId, runId }) => {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [plots, setPlots] = useState<Array<{
        title: string;
        x: number[];
        y: number[];
    }>>([]);

    useEffect(() => {
        const fetchAndProcessEvents = async () => {
            try {
                const events = await api.getRunEvents(orgId, runId);

                // Extract all metrics from events
                const metricsData: Record<string, { step: number; value: number }[]> = {};

                // Process each event
                events.forEach((event: Event) => {
                    const eventData = event.data;
                    const step = eventData.step || eventData.global_step;

                    if (step !== undefined) {
                        // For each metric in the event
                        Object.entries(eventData).forEach(([key, value]) => {
                            // Skip step/global_step keys and non-numeric values
                            if (
                                key !== 'step' &&
                                key !== 'global_step' &&
                                (typeof value === 'number' || (typeof value === 'string' && !isNaN(Number(value))))
                            ) {
                                if (!metricsData[key]) {
                                    metricsData[key] = [];
                                }

                                metricsData[key].push({
                                    step: Number(step),
                                    value: Number(value)
                                });
                            }
                        });
                    }
                });

                // Create plots for each metric
                const newPlots = [];
                for (const [metricName, dataPoints] of Object.entries(metricsData)) {
                    if (dataPoints.length > 1) { // Need at least 2 points for a line
                        // Sort by step to ensure correct line plotting
                        dataPoints.sort((a, b) => a.step - b.step);

                        newPlots.push({
                            title: metricName,
                            x: dataPoints.map(point => point.step),
                            y: dataPoints.map(point => point.value)
                        });
                    }
                }

                setPlots(newPlots);
                setLoading(false);
            } catch (error) {
                console.error('Error fetching events:', error);
                setError(error instanceof Error ? error.message : 'Error fetching metrics');
                setLoading(false);
            }
        };

        fetchAndProcessEvents();
    }, [orgId, runId]);

    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
            </Box>
        );
    }

    if (error) {
        return (
            <Box sx={{ p: 2 }}>
                <Typography color="error">Error loading metrics: {error}</Typography>
            </Box>
        );
    }

    if (plots.length === 0) {
        return (
            <Box sx={{ p: 2 }}>
                <Typography>No metrics data available for plotting</Typography>
            </Box>
        );
    }

    return (
        <Box>
            <Typography variant="h6" sx={{ mb: 2 }}>Metrics:</Typography>
            <Box sx={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
                gap: 3
            }}>
                {plots.map((plot, index) => (
                    <Box key={index} sx={{ minHeight: 400 }}>
                        <Plot
                            data={[
                                {
                                    x: plot.x,
                                    y: plot.y,
                                    type: 'scatter',
                                    mode: 'lines+markers',
                                    name: plot.title,
                                    line: {
                                        color: '#1f77b4', // Match the blue color from the example
                                        width: 2
                                    },
                                    marker: {
                                        size: 6
                                    }
                                }
                            ]}
                            layout={{
                                title: {
                                    text: plot.title,
                                    y: 0.95,
                                    x: 0.05,
                                    xanchor: 'left',
                                    yanchor: 'top',
                                    font: {
                                        size: 16,
                                        color: '#333'
                                    }
                                },
                                xaxis: {
                                    title: 'step',
                                    showgrid: true,
                                    gridcolor: '#E1E5EA',
                                    zeroline: false
                                },
                                yaxis: {
                                    title: plot.title,
                                    showgrid: true,
                                    gridcolor: '#E1E5EA',
                                    zeroline: false
                                },
                                autosize: true,
                                margin: {
                                    t: 60,
                                    r: 30,
                                    l: 60,
                                    b: 50
                                },
                                plot_bgcolor: 'white',
                                paper_bgcolor: 'white',
                                showlegend: true,
                                legend: {
                                    x: 1,
                                    y: 1,
                                    xanchor: 'right',
                                    yanchor: 'top',
                                    bgcolor: 'rgba(255, 255, 255, 0.8)',
                                    bordercolor: '#E1E5EA',
                                    borderwidth: 1
                                }
                            }}
                            style={{ width: '100%', height: '100%' }}
                            config={{
                                responsive: true,
                                displayModeBar: true,
                                displaylogo: false,
                                modeBarButtonsToAdd: ['toImage'],
                                modeBarButtonsToRemove: ['select2d', 'lasso2d'],
                                toImageButtonOptions: {
                                    format: 'png',
                                    filename: `${plot.title}_${runId}`,
                                    height: 800,
                                    width: 1200,
                                    scale: 2
                                }
                            }}
                        />
                    </Box>
                ))}
            </Box>
        </Box>
    );
};
