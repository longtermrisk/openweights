import { useEffect, useState } from 'react';
import { Alert, Box, Button, Checkbox, FormControlLabel, MenuItem, Paper, Stack, Table, TableBody, TableCell, TableHead, TableRow, TablePagination, TextField, Typography } from '@mui/material';
import { Link } from 'react-router-dom';
import { api } from '../api';
import { useOrganization } from '../contexts/OrganizationContext';

interface Amounts { direct_usd: number | null; overhead_usd: number | null; total_usd: number | null }
export interface CostReport extends Amounts {
    as_of: string; job_count: number; worker_count: number; unallocated_overhead_usd: number; unknown_runs: number; unpriced_workers: number;
    can_manage_limits: boolean;
    jobs: (Amounts & { job_id: string; unknown_runs: number })[];
    workers: (Amounts & { worker_id: string; hardware_type: string; rate_source: string; hourly_cost_usd: number | null })[];
    api_keys: (Amounts & { api_token_id: string | null; name: string | null })[];
    users: (Amounts & { user_id: string | null })[];
    limits: { api_token_id: string; limit_usd: number }[];
}
const usd = (value: number | null) => value == null ? 'Unknown' : new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 4 }).format(value);

export function CostsView() {
    const { currentOrganization } = useOrganization();
    const orgId = currentOrganization?.id;
    const [report, setReport] = useState<CostReport | null>(null);
    const [tokens, setTokens] = useState<{ id: string; name: string }[]>([]);
    const [error, setError] = useState('');
    const [refresh, setRefresh] = useState(0);
    const [includeOverhead, setIncludeOverhead] = useState(true);
    const [group, setGroup] = useState('jobs');
    const [page, setPage] = useState(0);
    const [keyId, setKeyId] = useState('');
    const [limit, setLimit] = useState('');
    const [saving, setSaving] = useState(false);
    useEffect(() => {
        setReport(null); setPage(0); setTokens([]); setKeyId(''); setLimit('');
    }, [orgId]);
    useEffect(() => {
        if (!orgId) return;
        let active = true;
        const load = async () => {
            try {
                const data = await api.getCosts(orgId, page * 100);
                const keys = data.can_manage_limits ? await api.listTokens(orgId) : [];
                if (active) { setReport(data); setTokens(keys); setError(''); }
            } catch (err) { if (active) setError(err instanceof Error ? err.message : 'Unable to load costs'); }
        };
        void load();
        const timer = setInterval(load, 30000);
        return () => { active = false; clearInterval(timer); };
    }, [orgId, refresh, page]);
    const saveLimit = async () => {
        if (!orgId || !keyId) return;
        const amount = limit.trim() === '' ? null : Number(limit);
        if (amount !== null && (!Number.isFinite(amount) || amount < 0)) { setError('Enter a nonnegative USD amount, or leave blank for unlimited.'); return; }
        setSaving(true);
        try { await api.setCostLimit(orgId, keyId, amount); setRefresh(n => n + 1); }
        catch (err) { setError(err instanceof Error ? err.message : 'Unable to save limit'); }
        finally { setSaving(false); }
    };
    const rows = !report ? [] : group === 'jobs' ? report.jobs.map(r => ({ ...r, id: r.job_id, label: r.job_id, link: `/${orgId}/jobs/${r.job_id}` }))
        : group === 'workers' ? report.workers.map(r => ({ ...r, id: r.worker_id, label: `${r.worker_id} · ${r.hardware_type || 'Unknown hardware'} · ${r.rate_source || 'unknown'} (${usd(r.hourly_cost_usd)}/hr)`, link: `/${orgId}/workers/${r.worker_id}` }))
        : group === 'api_keys' ? report.api_keys.map(r => ({ ...r, id: r.api_token_id || 'unknown', label: r.name ? `${r.name} (${r.api_token_id})` : r.api_token_id || 'Unattributed', link: '' }))
        : report.users.map(r => ({ ...r, id: r.user_id || 'unknown', label: r.user_id || 'Unattributed', link: '' }));
    return <Stack spacing={2}>
        <Stack direction="row" justifyContent="space-between"><Typography variant="h5">Compute costs</Typography><Button onClick={() => setRefresh(n => n + 1)}>Refresh</Button></Stack>
        {error && <Alert severity="error">{error}</Alert>}
        {!report ? <Typography>Loading costs…</Typography> : <>
            <Typography color="text.secondary">Lifetime USD estimates · updated {new Date(report.as_of).toLocaleString()} · refreshes every 30 seconds</Typography>
            {(report.unknown_runs > 0 || report.unpriced_workers > 0) && <Alert severity="warning">Totals are incomplete: {report.unknown_runs} runs and {report.unpriced_workers} workers have no recorded price. Historical usage is not priced retroactively.</Alert>}
            <Stack direction="row" spacing={3} flexWrap="wrap">
                {[['Total', report.total_usd], ['Job execution', report.direct_usd], ['Startup and idle overhead', report.overhead_usd], ['Overhead without jobs', report.unallocated_overhead_usd]].map(([label, value]) => <Paper key={String(label)} sx={{ p: 2 }}><Typography>{label}</Typography><Typography variant="h5">{usd(value as number)}</Typography></Paper>)}
            </Stack>
            <Typography variant="body2">Each worker’s overhead is divided equally among its distinct jobs, including retries. Allocations change while workers are active. Workers that never run a job retain unallocated overhead.</Typography>
            <Stack direction="row" spacing={2}>
                <TextField select label="Group by" value={group} onChange={e => { setGroup(e.target.value); setPage(0); }} sx={{ minWidth: 180 }}>
                    {[['jobs', 'Job'], ['workers', 'Worker'], ['api_keys', 'API key'], ['users', 'User']].map(([value, label]) => <MenuItem key={value} value={value}>{label}</MenuItem>)}
                </TextField>
                <FormControlLabel control={<Checkbox checked={includeOverhead} onChange={e => setIncludeOverhead(e.target.checked)} />} label="Include allocated overhead" />
            </Stack>
            <Box sx={{ overflowX: 'auto' }}><Table size="small"><TableHead><TableRow><TableCell>{group === 'api_keys' ? 'API key' : group}</TableCell><TableCell align="right">Execution</TableCell><TableCell align="right">Overhead</TableCell><TableCell align="right">{includeOverhead ? 'Total' : 'Execution only'}</TableCell></TableRow></TableHead><TableBody>
                {rows.map(row => <TableRow key={row.id}><TableCell>{row.link ? <Link to={row.link}>{row.label}</Link> : row.label}</TableCell><TableCell align="right">{usd(row.direct_usd)}</TableCell><TableCell align="right">{usd(row.overhead_usd)}</TableCell><TableCell align="right">{usd(includeOverhead ? row.total_usd : row.direct_usd)}</TableCell></TableRow>)}
                {!rows.length && <TableRow><TableCell colSpan={4}>No recorded usage yet.</TableCell></TableRow>}
            </TableBody></Table></Box>
            {(group === 'jobs' || group === 'workers') && <TablePagination component="div" count={group === 'jobs' ? report.job_count : report.worker_count} page={page} rowsPerPage={100} rowsPerPageOptions={[100]} onPageChange={(_, value) => setPage(value)} />}
            <Typography variant="h6">API key spending limits</Typography>
            <Typography variant="body2">Lifetime limits include allocated overhead. Once reached, new work is blocked and queued/running work is canceled on the next manager check. Concurrent jobs, shutdown time and later overhead can exceed the limit. These are not exact invoice caps.</Typography>
            {report.limits.map(l => <Typography key={l.api_token_id}>{tokens.find(t => t.id === l.api_token_id)?.name || l.api_token_id}: {usd(l.limit_usd)} limit · {usd(report.api_keys.find(k => k.api_token_id === l.api_token_id)?.total_usd ?? 0)} spent</Typography>)}
            {!report.limits.length && <Typography color="text.secondary">No limits configured.</Typography>}
            {report.can_manage_limits ? <Stack direction="row" spacing={2}>
                <TextField select label="API key" value={keyId} sx={{ minWidth: 220 }} disabled={saving} onChange={e => { setKeyId(e.target.value); const amount = report.limits.find(l => l.api_token_id === e.target.value)?.limit_usd; setLimit(amount == null ? '' : String(amount)); }}>
                    {tokens.map(t => <MenuItem key={t.id} value={t.id}>{t.name} ({t.id.slice(0, 8)})</MenuItem>)}
                </TextField>
                <TextField label="Lifetime USD limit" type="number" value={limit} disabled={saving} onChange={e => setLimit(e.target.value)} helperText="Blank removes limit; 0 blocks work" inputProps={{ min: 0, step: 'any' }} />
                <Button disabled={!keyId || saving} onClick={saveLimit}>Save limit</Button>
            </Stack> : <Typography color="text.secondary">Sign in as an organization admin to change limits.</Typography>}
        </>}
    </Stack>;
}
