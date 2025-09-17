import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
    Box,
    Typography,
    Slider,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Paper,
    Dialog,
    DialogTitle,
    DialogContent,
    CircularProgress,
    Switch,
    FormControlLabel,
} from '@mui/material';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

interface LogProbEvent {
    type: 'logprobs';
    loss: number;
    step: number;
    file: string;
}

interface Token {
    token: string;
    token_id: number;
    logp: number;
}

interface LogProbData {
    messages: any[];
    tokens: Token[];
}

interface TokenHistory {
    steps: number[];
    logprobs: number[];
}

interface Props {
    events: LogProbEvent[];
    getFileContent: (fileId: string) => Promise<string>;
}

export const LogProbVisualization: React.FC<Props> = ({ events, getFileContent }) => {
    const [selectedDataset, setSelectedDataset] = useState<string>('');
    const [sequenceIndex, setSequenceIndex] = useState<number>(0);
    const [step, setStep] = useState<number>(0);
    const [logProbData, setLogProbData] = useState<{ [key: string]: { [step: number]: LogProbData[] } }>({});
    const [selectedToken, setSelectedToken] = useState<{ token: string; tokenId: number; position: number } | null>(null);
    const [dialogOpen, setDialogOpen] = useState(false);
    const [tokenHistory, setTokenHistory] = useState<TokenHistory>({ steps: [], logprobs: [] });
    const [loadingHistory, setLoadingHistory] = useState(false);
    const [showSlider, setShowSlider] = useState(true);
    const [useLogProbColors, setUseLogProbColors] = useState(false);

    // Refs for intersection observer
    const contentRef = useRef<HTMLDivElement>(null);
    const controlsRef = useRef<HTMLDivElement>(null);

    // Extract unique datasets and steps from events
    const datasets = useMemo(() => {
        const datasetSet = new Set<string>();
        events.forEach(event => {
            const datasetKey = Object.keys(event).find(key =>
                key !== 'type' && key !== 'loss' && key !== 'step' && key !== 'file'
            );
            if (datasetKey) {
                datasetSet.add(datasetKey);
            }
        });
        return Array.from(datasetSet);
    }, [events]);

    const steps = useMemo(() => {
        return events
            .filter(e => e.type === 'logprobs')
            .map(e => e.step)
            .sort((a, b) => a - b);
    }, [events]);

    // Load log prob data for a specific step and dataset
    const loadLogProbData = async (event: LogProbEvent) => {
        try {
            const content = await getFileContent(event.file);
            const data = JSON.parse(content) as LogProbData[];

            const datasetKey = Object.keys(event).find(key =>
                key !== 'type' && key !== 'loss' && key !== 'step' && key !== 'file'
            ) || '';

            setLogProbData(prev => ({
                ...prev,
                [datasetKey]: {
                    ...(prev[datasetKey] || {}),
                    [event.step]: data
                }
            }));
        } catch (error) {
            console.error('Error loading log prob data:', error);
        }
    };

    // Eager load all data for the selected dataset
    useEffect(() => {
        if (!selectedDataset) return;

        const relevantEvents = events.filter(e => {
            const datasetKey = Object.keys(e).find(key =>
                key !== 'type' && key !== 'loss' && key !== 'step' && key !== 'file'
            );
            return datasetKey === selectedDataset;
        });

        // Load all data in parallel
        relevantEvents.forEach(event => {
            if (!logProbData[selectedDataset]?.[event.step]) {
                loadLogProbData(event);
            }
        });
    }, [selectedDataset]);

    // Initialize selected dataset and step
    useEffect(() => {
        if (datasets.length > 0 && !selectedDataset) {
            setSelectedDataset(datasets[0]);
        }
        if (steps.length > 0 && !step) {
            setStep(steps[0]);
        }
    }, [datasets, steps]);

    // Set up intersection observer for content visibility
    useEffect(() => {
        if (!contentRef.current) return;

        const options = {
            root: null,
            rootMargin: '0px',
            threshold: 0.1, // 10% visibility threshold
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                setShowSlider(entry.isIntersecting);
            });
        }, options);

        observer.observe(contentRef.current);

        return () => observer.disconnect();
    }, []);

    // Load token history when a token is selected
    useEffect(() => {
        const loadTokenHistory = async () => {
            if (!selectedToken || !selectedDataset) return;

            setLoadingHistory(true);
            const history: TokenHistory = { steps: [], logprobs: [] };

            try {
                // Get all available steps for the selected dataset
                const availableSteps = Object.keys(logProbData[selectedDataset] || {})
                    .map(Number)
                    .sort((a, b) => a - b);

                // For each step, get the token at the same position
                availableSteps.forEach(stepNum => {
                    const sequences = logProbData[selectedDataset][stepNum];
                    if (!sequences || !sequences[sequenceIndex]) return;

                    const sequence = sequences[sequenceIndex];
                    // Get the token at the same position as the selected token
                    if (selectedToken.position < sequence.tokens.length) {
                        const tokenAtPosition = sequence.tokens[selectedToken.position];
                        history.steps.push(stepNum);
                        history.logprobs.push(tokenAtPosition.logp);
                    }
                });

                // Sort by steps (should already be sorted, but just to be safe)
                const sorted = history.steps.map((step, i) => ({ step, logp: history.logprobs[i] }))
                    .sort((a, b) => a.step - b.step);

                setTokenHistory({
                    steps: sorted.map(s => s.step),
                    logprobs: sorted.map(s => s.logp)
                });
            } finally {
                setLoadingHistory(false);
            }
        };

        if (dialogOpen && selectedToken) {
            loadTokenHistory();
        }
    }, [dialogOpen, selectedToken, selectedDataset, logProbData, sequenceIndex]);

    const getColorFromLogP = (logp: number): string => {
        if (useLogProbColors) {
            // Direct logprob visualization (range: -20 to 0)
            const normalizedValue = Math.max(0, Math.min(1, (logp + 20) / 20));
            return `rgba(255, ${Math.floor(255 * normalizedValue)}, ${Math.floor(255 * normalizedValue)}, 0.3)`;
        } else {
            // Probability-based visualization
            const prob = Math.exp(logp);
            return `rgba(255, ${Math.floor(255 * prob)}, ${Math.floor(255 * prob)}, 0.3)`;
        }
    };

    const currentData = selectedDataset && step && logProbData[selectedDataset]?.[step]?.[sequenceIndex];
    const maxSequences = selectedDataset && step && logProbData[selectedDataset]?.[step]?.length || 0;

    const renderTokens = (tokens: Token[]) => {
        return tokens.map((token, position) => {
            const containsNewline = token.token.includes('\n');

            if (containsNewline) {
                const parts = token.token.split(/(\n)/);
                return (
                    <React.Fragment key={position}>
                        {parts.map((part, j) => (
                            part === '\n' ? (
                                <Box key={`${position}-${j}`} sx={{ width: '100%', height: 0 }} />
                            ) : (
                                part && (
                                    <Box
                                        key={`${position}-${j}`}
                                        onClick={() => {
                                            setSelectedToken({
                                                token: token.token,
                                                tokenId: token.token_id,
                                                position: position
                                            });
                                            setDialogOpen(true);
                                        }}
                                        sx={{
                                            cursor: 'pointer',
                                            p: 0.5,
                                            borderRadius: 1,
                                            backgroundColor: getColorFromLogP(token.logp),
                                            '&:hover': {
                                                outline: '1px solid blue'
                                            }
                                        }}
                                    >
                                        <Typography variant="body2" component="span">
                                            {part}
                                        </Typography>
                                    </Box>
                                )
                            )
                        ))}
                    </React.Fragment>
                );
            }

            return (
                <Box
                    key={position}
                    onClick={() => {
                        setSelectedToken({
                            token: token.token,
                            tokenId: token.token_id,
                            position: position
                        });
                        setDialogOpen(true);
                    }}
                    sx={{
                        cursor: 'pointer',
                        p: 0.5,
                        borderRadius: 1,
                        backgroundColor: getColorFromLogP(token.logp),
                        '&:hover': {
                            outline: '1px solid blue'
                        }
                    }}
                >
                    <Typography variant="body2" component="span">
                        {token.token}
                    </Typography>
                </Box>
            );
        });
    };

    return (
        <Box sx={{ mt: 2 }}>
            {/* Controls */}
            <Box
                ref={controlsRef}
                sx={{
                    position: showSlider ? 'sticky' : 'static',
                    top: 0,
                    zIndex: 1,
                    backgroundColor: 'white',
                    pb: 2,
                    borderBottom: showSlider ? '1px solid rgba(0, 0, 0, 0.12)' : 'none',
                }}
            >
                <Box sx={{ display: 'flex', gap: 2, mb: 2, alignItems: 'center' }}>
                    <FormControl sx={{ minWidth: 200 }}>
                        <InputLabel>Dataset</InputLabel>
                        <Select
                            value={selectedDataset}
                            onChange={(e) => setSelectedDataset(e.target.value)}
                            label="Dataset"
                        >
                            {datasets.map(dataset => (
                                <MenuItem key={dataset} value={dataset}>{dataset}</MenuItem>
                            ))}
                        </Select>
                    </FormControl>

                    <FormControl sx={{ minWidth: 200 }}>
                        <InputLabel>Sequence Index</InputLabel>
                        <Select
                            value={sequenceIndex}
                            onChange={(e) => setSequenceIndex(Number(e.target.value))}
                            label="Sequence Index"
                        >
                            {[...Array(maxSequences)].map((_, i) => (
                                <MenuItem key={i} value={i}>{i}</MenuItem>
                            ))}
                        </Select>
                    </FormControl>

                    <FormControlLabel
                        control={
                            <Switch
                                checked={useLogProbColors}
                                onChange={(e) => setUseLogProbColors(e.target.checked)}
                            />
                        }
                        label="Use LogProb Colors"
                    />
                </Box>

                <Box sx={{ px: 2 }}>
                    <Typography gutterBottom>Step: {step}</Typography>
                    <Slider
                        value={step}
                        onChange={(_, value) => setStep(value as number)}
                        min={Math.min(...steps)}
                        max={Math.max(...steps)}
                        step={null}
                        marks={steps.map(s => ({ value: s, label: s.toString() }))}
                    />
                </Box>
            </Box>

            {/* Token visualization */}
            <Box ref={contentRef}>
                {currentData ? (
                    <Paper sx={{ p: 2, mt: 2 }}>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                            {renderTokens(currentData.tokens)}
                        </Box>
                    </Paper>
                ) : (
                    <Typography>No data available for the selected parameters</Typography>
                )}
            </Box>

            {/* Token history dialog */}
            <Dialog
                open={dialogOpen}
                onClose={() => setDialogOpen(false)}
                maxWidth="md"
                fullWidth
            >
                <DialogTitle>
                    Token History: {selectedToken?.token} (Position: {selectedToken?.position})
                </DialogTitle>
                <DialogContent>
                    {loadingHistory ? (
                        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                            <CircularProgress />
                        </Box>
                    ) : selectedToken && (
                        <Box sx={{ height: 400 }}>
                            <Line
                                data={{
                                    labels: tokenHistory.steps,
                                    datasets: [{
                                        label: 'Log Probability',
                                        data: tokenHistory.logprobs,
                                        borderColor: 'rgb(75, 192, 192)',
                                        tension: 0.1
                                    }]
                                }}
                                options={{
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    plugins: {
                                        title: {
                                            display: true,
                                            text: `Log Probability Evolution for Token: ${selectedToken.token} at Position ${selectedToken.position}`
                                        }
                                    },
                                    scales: {
                                        y: {
                                            title: {
                                                display: true,
                                                text: 'Log Probability'
                                            }
                                        },
                                        x: {
                                            title: {
                                                display: true,
                                                text: 'Training Step'
                                            }
                                        }
                                    }
                                }}
                            />
                        </Box>
                    )}
                </DialogContent>
            </Dialog>
        </Box>
    );
};
