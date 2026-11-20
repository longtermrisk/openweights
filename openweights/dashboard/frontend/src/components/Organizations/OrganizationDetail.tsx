import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Breadcrumbs,
  Alert,
  Tooltip,
} from '@mui/material'
import DeleteIcon from '@mui/icons-material/Delete'
import EditIcon from '@mui/icons-material/Edit'
import NavigateNextIcon from '@mui/icons-material/NavigateNext'
import { useAuth } from '../../contexts/AuthContext'
import { TokenView } from '../TokenView'
import { api, OrganizationSecret } from '../../api'

interface Member {
  user_id: string;
  email?: string;
  role: 'admin' | 'user';
}

interface Organization {
  id: string
  name: string
}

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

// Required secrets that cannot be deleted
const REQUIRED_SECRETS = ['HF_ORG', 'HF_USER', 'HF_TOKEN', 'RUNPOD_API_KEY', 'OPENWEIGHTS_API_KEY'];
// Settings that are not secrets; the backend returns these in full.
const PLAIN_SETTINGS = ['HF_USER', 'HF_ORG', 'OW_MANAGED', 'MAX_WORKERS'];

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  )
}

export function OrganizationDetail() {
  const { orgId } = useParams<{ orgId: string }>()
  const { user } = useAuth()
  const [organization, setOrganization] = useState<Organization | null>(null)
  const [members, setMembers] = useState<Member[]>([])
  const [secrets, setSecrets] = useState<OrganizationSecret[]>([])
  const [isAdmin, setIsAdmin] = useState(false)
  const [inviteEmail, setInviteEmail] = useState('')
  const [openInviteDialog, setOpenInviteDialog] = useState(false)
  const [openEditDialog, setOpenEditDialog] = useState(false)
  // The secret dialog adds a secret (name === null) or replaces the value of one.
  const [secretDialog, setSecretDialog] = useState<{ name: string | null } | null>(null)
  const [savingSecret, setSavingSecret] = useState(false)
  const [editName, setEditName] = useState('')
  const [newSecretName, setNewSecretName] = useState('')
  const [newSecretValue, setNewSecretValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [secretError, setSecretError] = useState<string | null>(null)
  const [tabValue, setTabValue] = useState(0)

  useEffect(() => {
    if (orgId) {
      fetchOrganizationData()
    }
  }, [orgId])

  async function fetchOrganizationData() {
    if (!orgId) return;
    try {
      const org = await api.getOrganization(orgId);
      setOrganization(org);
      setEditName(org.name);

      const memberData = await api.listOrganizationMembers(orgId);
      const currentUserMember = memberData.find(m => m.user_id === user?.id);
      const admin = currentUserMember?.role === 'admin';
      setIsAdmin(admin);
      setMembers(memberData.map(m => ({
        user_id: m.user_id,
        role: m.role,
        ...(m.email ? { email: m.email } : {})
      })));

      if (admin) {
        setSecrets(await api.listOrganizationSecrets(orgId));
      }
    } catch (err) {
      console.error('Error fetching data:', err);
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  const handleRoleChange = async (userId: string, newRole: 'admin' | 'user') => {
    if (!orgId) return;
    try {
      await api.updateOrganizationMemberRole(orgId, userId, newRole);
      setMembers(members.map(member =>
        member.user_id === userId ? { ...member, role: newRole } : member
      ));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update role');
    }
  }

  const handleRemoveMember = async (userId: string) => {
    if (!orgId) return;
    try {
      await api.removeOrganizationMember(orgId, userId);
      setMembers(members.filter(member => member.user_id !== userId));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove member');
    }
  }

  const handleInvite = async () => {
    if (!orgId) return;
    try {
      const data = await api.inviteOrganizationMember(orgId, inviteEmail, 'user');
      setMembers([...members, {
        user_id: data.user_id,
        email: data.email,
        role: data.role,
      }]);
      setInviteEmail('');
      setOpenInviteDialog(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send invitation');
    }
  }

  const handleUpdateOrganization = async () => {
    if (!orgId) return;
    try {
      const updated = await api.updateOrganization(orgId, editName);
      setOrganization(prev => prev ? { ...prev, name: updated.name } : updated);
      setOpenEditDialog(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update organization');
    }
  }

  const closeSecretDialog = () => {
    setSecretDialog(null);
    setNewSecretName('');
    setNewSecretValue('');
  };

  const handleSaveSecret = async () => {
    const name = secretDialog?.name ?? newSecretName.trim();
    if (!orgId || !name || !newSecretValue) return;
    try {
      setSecretError(null);
      setSavingSecret(true);
      await api.setOrganizationSecret(orgId, name, newSecretValue);
      closeSecretDialog();
      setSecrets(await api.listOrganizationSecrets(orgId));
    } catch (err) {
      closeSecretDialog();
      setSecretError(err instanceof Error ? err.message : 'Failed to save secret');
    } finally {
      setSavingSecret(false);
    }
  };

  const handleDeleteSecret = async (name: string) => {
    if (!orgId || REQUIRED_SECRETS.includes(name)) return;
    if (!window.confirm(`Delete ${name}? This cannot be undone.`)) return;
    try {
      setSecretError(null);
      await api.deleteOrganizationSecret(orgId, name);
      setSecrets(await api.listOrganizationSecrets(orgId));
    } catch (err) {
      setSecretError(err instanceof Error ? err.message : 'Failed to delete secret');
    }
  };

  if (error) return <Typography color="error">{error}</Typography>
  if (!organization) return <Typography>Loading...</Typography>
  if (!orgId) {
    return <Typography>Organization not found</Typography>;
  }

  return (
    <Box>
      <Breadcrumbs
        separator={<NavigateNextIcon fontSize="small" />}
        sx={{ mb: 3 }}
      >
        <Link to="/organizations" style={{ textDecoration: 'none', color: 'inherit' }}>
          Organizations
        </Link>
        <Link to={`/${orgId}/jobs`} style={{ textDecoration: 'none', color: 'inherit' }}>
          {organization.name}
        </Link>
        <Typography color="text.primary">Settings</Typography>
      </Breadcrumbs>

      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">
          Organization Settings
        </Typography>
        {isAdmin && (
          <IconButton onClick={() => setOpenEditDialog(true)}>
            <EditIcon />
          </IconButton>
        )}
      </Box>

      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
          <Tab label="Members" />
          {isAdmin && <Tab label="Secrets" />}
          {isAdmin && <Tab label="API Tokens" />}
        </Tabs>
      </Box>

      <TabPanel value={tabValue} index={0}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">Members</Typography>
          {isAdmin && (
            <Button
              variant="contained"
              color="primary"
              onClick={() => setOpenInviteDialog(true)}
            >
              Invite Member
            </Button>
          )}
        </Box>

        <List>
          {members.map((member) => (
            <ListItem key={member.user_id} divider>
              <ListItemText
                primary={member.email}
                secondary={member.user_id === user?.id ? '(You)' : null}
              />
              <Box display="flex" alignItems="center" gap={2}>
                {isAdmin && member.user_id !== user?.id && (
                  <FormControl variant="outlined" size="small">
                    <Select
                      value={member.role}
                      onChange={(e) => handleRoleChange(member.user_id, e.target.value as 'admin' | 'user')}
                    >
                      <MenuItem value="user">User</MenuItem>
                      <MenuItem value="admin">Admin</MenuItem>
                    </Select>
                  </FormControl>
                )}
                {!isAdmin && (
                  <Typography variant="body2" color="textSecondary">
                    {member.role}
                  </Typography>
                )}
                {isAdmin && member.user_id !== user?.id && (
                  <IconButton
                    edge="end"
                    onClick={() => handleRemoveMember(member.user_id)}
                    color="error"
                  >
                    <DeleteIcon />
                  </IconButton>
                )}
              </Box>
            </ListItem>
          ))}
        </List>
      </TabPanel>

      {isAdmin && (
        <TabPanel value={tabValue} index={1}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">Secrets</Typography>
            <Button
              variant="contained"
              color="primary"
              onClick={() => setSecretDialog({ name: null })}
            >
              Add Secret
            </Button>
          </Box>

          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Stored secrets cannot be viewed, only replaced or deleted. A secret is
            identified by its last four characters.
          </Typography>

          {secretError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {secretError}
            </Alert>
          )}

          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Value</TableCell>
                  <TableCell>Last changed</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {secrets.map((secret) => (
                  <TableRow key={secret.name}>
                    <TableCell>{secret.name}</TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>
                      {secret.plain_value ?? `••••••••${secret.last_four ?? ''}`}
                    </TableCell>
                    <TableCell>
                      {secret.updated_at ? new Date(secret.updated_at).toLocaleString() : ''}
                    </TableCell>
                    <TableCell align="right">
                      <Button
                        size="small"
                        onClick={() => setSecretDialog({ name: secret.name })}
                      >
                        Replace
                      </Button>
                      {!REQUIRED_SECRETS.includes(secret.name) && (
                        <Tooltip title="Delete">
                          <IconButton
                            onClick={() => handleDeleteSecret(secret.name)}
                            color="error"
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>
      )}

      {isAdmin && (
        <TabPanel value={tabValue} index={2}>
          <TokenView orgId={orgId} />
        </TabPanel>
      )}

      {/* Invite Dialog */}
      <Dialog open={openInviteDialog} onClose={() => setOpenInviteDialog(false)}>
        <DialogTitle>Invite New Member</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Email Address"
            type="email"
            fullWidth
            value={inviteEmail}
            onChange={(e) => setInviteEmail(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenInviteDialog(false)}>Cancel</Button>
          <Button
            onClick={handleInvite}
            variant="contained"
            color="primary"
            disabled={!inviteEmail}
          >
            Send Invitation
          </Button>
        </DialogActions>
      </Dialog>

      {/* Edit Organization Dialog */}
      <Dialog open={openEditDialog} onClose={() => setOpenEditDialog(false)}>
        <DialogTitle>Edit Organization</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Organization Name"
            type="text"
            fullWidth
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenEditDialog(false)}>Cancel</Button>
          <Button
            onClick={handleUpdateOrganization}
            variant="contained"
            color="primary"
            disabled={!editName || editName === organization.name}
          >
            Update
          </Button>
        </DialogActions>
      </Dialog>

      {/* Add or replace a secret. The value is write-only. */}
      <Dialog open={secretDialog !== null} onClose={closeSecretDialog}>
        <DialogTitle>
          {secretDialog?.name ? `Replace ${secretDialog.name}` : 'Add New Secret'}
        </DialogTitle>
        <DialogContent>
          {!secretDialog?.name && (
            <TextField
              autoFocus
              margin="dense"
              label="Secret Name"
              type="text"
              fullWidth
              value={newSecretName}
              onChange={(e) => setNewSecretName(e.target.value)}
              sx={{ mb: 2 }}
            />
          )}
          <TextField
            autoFocus={!!secretDialog?.name}
            margin="dense"
            label={secretDialog?.name ? 'New Value' : 'Secret Value'}
            type={secretDialog?.name && PLAIN_SETTINGS.includes(secretDialog.name) ? 'text' : 'password'}
            autoComplete="new-password"
            fullWidth
            value={newSecretValue}
            onChange={(e) => setNewSecretValue(e.target.value)}
            helperText="The value is stored when you save and cannot be shown again."
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={closeSecretDialog}>Cancel</Button>
          <Button
            onClick={handleSaveSecret}
            variant="contained"
            color="primary"
            disabled={savingSecret || !newSecretValue || !(secretDialog?.name || newSecretName.trim())}
          >
            {secretDialog?.name ? 'Replace' : 'Add Secret'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}
