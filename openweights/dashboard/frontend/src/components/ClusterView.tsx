import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Alert, Box, FormControlLabel, Paper, Switch, Typography } from '@mui/material';
import { api } from '../api';
import { RefreshButton } from './RefreshButton';

const REFRESH_INTERVAL_MS = 15000;
const TIMESTAMP = /^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d{3}\s+(?:- )?/;

interface LogLine {
    text: string;
    count: number;
    firstSeen?: string;
}

// The org manager logs the same few lines every loop (~15s). Keep only the latest
// occurrence of each message, ignoring numbers (timestamps, countdowns, counts).
function collapseRepeats(lines: string[]): LogLine[] {
    const byKey = new Map<string, LogLine & { index: number }>();
    lines.forEach((text, index) => {
        const timestamp = text.match(TIMESTAMP)?.[1];
        const key = text.replace(TIMESTAMP, '').replace(/\d+/g, '#');
        const previous = byKey.get(key);
        byKey.set(key, {
            text,
            index,
            count: (previous?.count ?? 0) + 1,
            firstSeen: previous?.firstSeen ?? timestamp,
        });
    });
    return [...byKey.values()].sort((a, b) => a.index - b.index);
}

function lineColor(text: string): string | undefined {
    if (/error|failed|exception|traceback|cannot start/i.test(text)) return '#b71c1c';
    if (/warn|cooldown|paused/i.test(text)) return '#e65100';
    return undefined;
}

export const ClusterView: React.FC = () => {
    const { orgId } = useParams<{ orgId: string }>();
    const [log, setLog] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [collapse, setCollapse] = useState(true);
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [lastRefresh, setLastRefresh] = useState<Date>();
    const scrollRef = useRef<HTMLDivElement>(null);

    const fetchLog = useCallback(async () => {
        if (!orgId) return;
        try {
            setLog(await api.getClusterLogs(orgId));
            setError(null);
            setLastRefresh(new Date());
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }, [orgId]);

    useEffect(() => {
        fetchLog();
        if (!autoRefresh) return;
        const interval = setInterval(fetchLog, REFRESH_INTERVAL_MS);
        return () => clearInterval(interval);
    }, [fetchLog, autoRefresh]);

    const lines = useMemo(() => (log ? log.split('\n').filter(Boolean) : []), [log]);
    const shown: LogLine[] = useMemo(
        () => (collapse ? collapseRepeats(lines) : lines.map(text => ({ text, count: 1 }))),
        [lines, collapse]
    );
    const status = useMemo(
        () => [...lines].reverse().find(line => /\] workers: /.test(line)),
        [lines]
    );

    useEffect(() => {
        const element = scrollRef.current;
        if (element) element.scrollTop = element.scrollHeight;
    }, [shown]);

    return (
        <Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1, flexWrap: 'wrap' }}>
                <Typography variant="h5">Cluster manager</Typography>
                <RefreshButton onRefresh={fetchLog} lastRefresh={lastRefresh} />
                <Box sx={{ flexGrow: 1 }} />
                <FormControlLabel
                    control={<Switch checked={collapse} onChange={e => setCollapse(e.target.checked)} />}
                    label="Collapse repeats"
                />
                <FormControlLabel
                    control={<Switch checked={autoRefresh} onChange={e => setAutoRefresh(e.target.checked)} />}
                    label="Auto-refresh"
                />
            </Box>
            <Typography color="text.secondary" sx={{ mb: 2 }}>
                Log of the process that starts and stops GPU workers for this organization. If a job
                stays pending, the reason is usually here.
            </Typography>
            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
            {status && (
                <Alert severity="info" sx={{ mb: 2, fontFamily: 'monospace' }}>
                    {status.replace(/\[org=[^\]]*\]\s*/, '')}
                </Alert>
            )}
            <Paper
                ref={scrollRef}
                variant="outlined"
                sx={{
                    p: 1.5,
                    maxHeight: '70vh',
                    overflow: 'auto',
                    fontFamily: 'monospace',
                    fontSize: '0.8rem',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                }}
            >
                {log === null && !error && <Typography>Loading…</Typography>}
                {log !== null && shown.length === 0 && (
                    <Typography color="text.secondary">
                        No cluster manager logs yet. They appear once the manager for this
                        organization has started.
                    </Typography>
                )}
                {shown.map((line, i) => (
                    <Box key={i} sx={{ color: lineColor(line.text) }}>
                        {line.text}
                        {line.count > 1 && (
                            <Box component="span" sx={{ color: 'text.secondary' }}>
                                {`  (×${line.count}${line.firstSeen ? `, first at ${line.firstSeen}` : ''})`}
                            </Box>
                        )}
                    </Box>
                ))}
            </Paper>
        </Box>
    );
};
