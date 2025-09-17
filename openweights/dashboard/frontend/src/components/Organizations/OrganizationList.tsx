import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    Card,
    CardContent,
    Typography,
    Grid,
    Button,
    Container,
    Box,
    CircularProgress,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    TextField,
    Alert,
} from '@mui/material';
import { useOrganization } from '../../contexts/OrganizationContext';
import { api } from '../../api';

interface CreateOrgFormData {
    name: string;
    HF_USER: string;
    HF_ORG: string;
    HF_TOKEN: string;
    RUNPOD_API_KEY: string;
}

const initialFormData: CreateOrgFormData = {
    name: '',
    HF_USER: '',
    HF_ORG: '',
    HF_TOKEN: '',
    RUNPOD_API_KEY: '',
};

export function OrganizationList() {
    const { organizations, setCurrentOrganization, loading, loadOrganizations } = useOrganization();
    const navigate = useNavigate();
    const [openCreateDialog, setOpenCreateDialog] = useState(false);
    const [formData, setFormData] = useState<CreateOrgFormData>(initialFormData);
    const [createError, setCreateError] = useState<string | null>(null);
    const [creating, setCreating] = useState(false);

    const handleInputChange = (field: keyof CreateOrgFormData) => (
        event: React.ChangeEvent<HTMLInputElement>
    ) => {
        setFormData(prev => ({
            ...prev,
            [field]: event.target.value
        }));
    };

    const handleCreate = async () => {
        setCreating(true);
        setCreateError(null);
        try {
            const response = await api.createOrganization({
                name: formData.name,
                secrets: {
                    HF_USER: formData.HF_USER,
                    HF_ORG: formData.HF_ORG,
                    HF_TOKEN: formData.HF_TOKEN,
                    RUNPOD_API_KEY: formData.RUNPOD_API_KEY,
                }
            });

            await loadOrganizations();
            setOpenCreateDialog(false);
            setFormData(initialFormData);
            navigate(`/${response.id}/jobs`);
        } catch (error) {
            setCreateError(error instanceof Error ? error.message : 'Failed to create organization');
        } finally {
            setCreating(false);
        }
    };

    const isFormValid = () => {
        return Object.values(formData).every(value => value.trim() !== '');
    };

    if (loading) {
        return (
            <Box
                display="flex"
                justifyContent="center"
                alignItems="center"
                minHeight="calc(100vh - 100px)"
            >
                <CircularProgress />
            </Box>
        );
    }

    return (
        <Container maxWidth="lg" sx={{ mt: 4 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
                <Typography variant="h4" gutterBottom>
                    Organizations
                </Typography>
                <Button
                    variant="contained"
                    color="primary"
                    onClick={() => setOpenCreateDialog(true)}
                >
                    Create Organization
                </Button>
            </Box>

            {organizations.length === 0 ? (
                <Typography color="text.secondary" align="center">
                    No organizations found. Create one to get started.
                </Typography>
            ) : (
                <Grid container spacing={3}>
                    {organizations.map(org => (
                        <Grid item xs={12} sm={6} md={4} key={org.id}>
                            <Card
                                sx={{
                                    height: '100%',
                                    display: 'flex',
                                    flexDirection: 'column',
                                    '&:hover': {
                                        boxShadow: 6
                                    }
                                }}
                            >
                                <CardContent sx={{ flexGrow: 1 }}>
                                    <Typography variant="h6" gutterBottom>
                                        {org.name}
                                    </Typography>
                                    <Typography color="text.secondary" variant="body2">
                                        Created: {new Date(org.created_at).toLocaleDateString()}
                                    </Typography>
                                    <Button
                                        variant="contained"
                                        onClick={() => {
                                            setCurrentOrganization(org);
                                            navigate(`/${org.id}/jobs`);
                                        }}
                                        sx={{ mt: 2 }}
                                        fullWidth
                                    >
                                        Select
                                    </Button>
                                </CardContent>
                            </Card>
                        </Grid>
                    ))}
                </Grid>
            )}

            <Dialog
                open={openCreateDialog}
                onClose={() => setOpenCreateDialog(false)}
                maxWidth="sm"
                fullWidth
            >
                <DialogTitle>Create New Organization</DialogTitle>
                <DialogContent>
                    {createError && (
                        <Alert severity="error" sx={{ mb: 2 }}>
                            {createError}
                        </Alert>
                    )}
                    <TextField
                        autoFocus
                        margin="dense"
                        label="Organization Name"
                        fullWidth
                        value={formData.name}
                        onChange={handleInputChange('name')}
                        sx={{ mb: 2 }}
                    />
                    <Typography variant="subtitle2" sx={{ mb: 2 }}>
                        Required Secrets
                    </Typography>
                    <TextField
                        margin="dense"
                        label="Hugging Face Username"
                        fullWidth
                        value={formData.HF_USER}
                        onChange={handleInputChange('HF_USER')}
                        sx={{ mb: 2 }}
                    />
                    <TextField
                        margin="dense"
                        label="Hugging Face Organization"
                        fullWidth
                        value={formData.HF_ORG}
                        onChange={handleInputChange('HF_ORG')}
                        sx={{ mb: 2 }}
                    />
                    <TextField
                        margin="dense"
                        label="Hugging Face Token"
                        fullWidth
                        type="password"
                        value={formData.HF_TOKEN}
                        onChange={handleInputChange('HF_TOKEN')}
                        sx={{ mb: 2 }}
                    />
                    <TextField
                        margin="dense"
                        label="RunPod API Key"
                        fullWidth
                        type="password"
                        value={formData.RUNPOD_API_KEY}
                        onChange={handleInputChange('RUNPOD_API_KEY')}
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setOpenCreateDialog(false)}>Cancel</Button>
                    <Button
                        onClick={handleCreate}
                        variant="contained"
                        disabled={!isFormValid() || creating}
                    >
                        {creating ? <CircularProgress size={24} /> : 'Create'}
                    </Button>
                </DialogActions>
            </Dialog>
        </Container>
    );
}
