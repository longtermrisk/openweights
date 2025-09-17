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
  InputAdornment,
  Tooltip,
} from '@mui/material'
import DeleteIcon from '@mui/icons-material/Delete'
import EditIcon from '@mui/icons-material/Edit'
import NavigateNextIcon from '@mui/icons-material/NavigateNext'
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import { supabase } from '../../supabaseClient'
import { useAuth } from '../../contexts/AuthContext'
import { TokenView } from '../TokenView'
import { api } from '../../api'

interface Member {
  user_id: string;
  email?: string;
  role: 'admin' | 'user';
}

interface Organization {
  id: string
  name: string
}

interface Secret {
  id: string
  name: string
  value: string
  created_at: string
  updated_at: string
}

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

// Required secrets that cannot be deleted
const REQUIRED_SECRETS = ['HF_ORG', 'HF_USER', 'HF_TOKEN', 'RUNPOD_API_KEY', 'OPENWEIGHTS_API_KEY'];

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
  const [editedSecrets, setEditedSecrets] = useState<Record<string, string>>({})
  const [isAdmin, setIsAdmin] = useState(false)
  const [inviteEmail, setInviteEmail] = useState('')
  const [openInviteDialog, setOpenInviteDialog] = useState(false)
  const [openEditDialog, setOpenEditDialog] = useState(false)
  const [openNewSecretDialog, setOpenNewSecretDialog] = useState(false)
  const [editName, setEditName] = useState('')
  const [newSecretName, setNewSecretName] = useState('')
  const [newSecretValue, setNewSecretValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [secretError, setSecretError] = useState<string | null>(null)
  const [tabValue, setTabValue] = useState(0)
  const [hasChanges, setHasChanges] = useState(false)
  const [showSecretValues, setShowSecretValues] = useState<Record<string, boolean>>({})

  useEffect(() => {
    if (orgId) {
      fetchOrganizationData()
    }
  }, [orgId])

  async function fetchOrganizationData() {
    try {
      // Fetch organization details
      const { data: org, error: orgError } = await supabase
        .from('organizations')
        .select('*')
        .eq('id', orgId)
        .single()

      if (orgError) throw orgError
      setOrganization(org)
      setEditName(org.name)

      // Fetch members
      const { data: memberData, error: memberError } = await supabase
        .rpc('get_organization_members', { org_id: orgId })

      if (memberError) throw memberError

      // Check if current user is admin
      const currentUserMember = memberData.find((m: { user_id: string }) => m.user_id === user?.id)
      setIsAdmin(currentUserMember?.role === 'admin')

      setMembers(memberData.map((m: { user_id: string; role: string; email?: string }) => ({
        user_id: m.user_id,
        role: m.role,
        ...(m.email ? { email: m.email } : {})
      })))

      // Fetch secrets if admin
      if (currentUserMember?.role === 'admin') {
        const { data: secretData, error: secretError } = await supabase
          .from('organization_secrets')
          .select('*')
          .eq('organization_id', orgId)

        if (secretError) throw secretError
        // Initialize edited secrets with current values
        const currentValues = Object.fromEntries(
          secretData.map((secret: Secret) => [secret.name, secret.value])
        );
        setEditedSecrets(currentValues);
      }
    } catch (err) {
      console.error('Error fetching data:', err)
      setError(err instanceof Error ? err.message : 'An error occurred')
    }
  }

  const handleRoleChange = async (userId: string, newRole: 'admin' | 'user') => {
    try {
      const { error } = await supabase
        .from('organization_members')
        .update({ role: newRole })
        .eq('organization_id', orgId)
        .eq('user_id', userId)

      if (error) throw error

      setMembers(members.map(member =>
        member.user_id === userId ? { ...member, role: newRole } : member
      ))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update role')
    }
  }

  const handleRemoveMember = async (userId: string) => {
    try {
      const { error } = await supabase
        .rpc('remove_organization_member', {
          org_id: orgId,
          member_id: userId
        })

      if (error) throw error

      setMembers(members.filter(member => member.user_id !== userId))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove member')
    }
  }

  const handleInvite = async () => {
    try {
      const { data, error } = await supabase
        .rpc('invite_organization_member', {
          org_id: orgId,
          member_email: inviteEmail,
          member_role: 'user'
        })

      if (error) throw error

      if (data) {
        setMembers([...members, {
          user_id: data.user_id,
          email: data.email,
          role: data.role
        }])
      }

      setInviteEmail('')
      setOpenInviteDialog(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send invitation')
    }
  }

  const handleUpdateOrganization = async () => {
    try {
      const { error } = await supabase
        .rpc('update_organization', {
          org_id: orgId,
          new_name: editName
        })

      if (error) throw error

      setOrganization(prev => prev ? { ...prev, name: editName } : null)
      setOpenEditDialog(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update organization')
    }
  }

  const handleSaveSecrets = async () => {
    if (!orgId) return;

    try {
      setSecretError(null);

      // Send all secrets to the backend
      await api.updateOrganizationSecrets(orgId, editedSecrets);

      // Refresh secrets
      const { data: secretData, error: secretError } = await supabase
        .from('organization_secrets')
        .select('*')
        .eq('organization_id', orgId);

      if (secretError) throw secretError;
      const currentValues = Object.fromEntries(
        secretData.map((secret: Secret) => [secret.name, secret.value])
      );
      setEditedSecrets(currentValues);
      setHasChanges(false);
      setSecretError(null);
    } catch (err) {
      console.error('Error updating secrets:', err);
      setSecretError(err instanceof Error ? err.message : 'Failed to update secrets');
    }
  };

  const handleDeleteSecret = (name: string) => {
    if (REQUIRED_SECRETS.includes(name)) return;

    const newSecrets = { ...editedSecrets };
    delete newSecrets[name];
    setEditedSecrets(newSecrets);
    setHasChanges(true);
  };

  const handleAddNewSecret = () => {
    if (!newSecretName || !newSecretValue) return;

    setEditedSecrets(prev => ({
      ...prev,
      [newSecretName]: newSecretValue
    }));
    setNewSecretName('');
    setNewSecretValue('');
    setOpenNewSecretDialog(false);
    setHasChanges(true);
  };

  const handleSecretChange = (name: string, value: string) => {
    setEditedSecrets(prev => ({
      ...prev,
      [name]: value
    }));
    setHasChanges(true);
  };

  const toggleSecretVisibility = (name: string) => {
    setShowSecretValues(prev => ({
      ...prev,
      [name]: !prev[name]
    }));
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
            <Box>
              <Button
                variant="contained"
                color="primary"
                onClick={() => setOpenNewSecretDialog(true)}
                sx={{ mr: 1 }}
              >
                Add Secret
              </Button>
              <Button
                variant="contained"
                color="primary"
                onClick={handleSaveSecrets}
                disabled={!hasChanges}
              >
                Save Changes
              </Button>
            </Box>
          </Box>

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
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(editedSecrets).map(([name, value]) => (
                  <TableRow key={name}>
                    <TableCell>{name}</TableCell>
                    <TableCell>
                      <TextField
                        fullWidth
                        type={showSecretValues[name] ? 'text' : 'password'}
                        value={value}
                        onChange={(e) => handleSecretChange(name, e.target.value)}
                        InputProps={{
                          endAdornment: (
                            <InputAdornment position="end">
                              <IconButton
                                onClick={() => toggleSecretVisibility(name)}
                                edge="end"
                              >
                                {showSecretValues[name] ? <VisibilityOffIcon /> : <VisibilityIcon />}
                              </IconButton>
                            </InputAdornment>
                          ),
                        }}
                      />
                    </TableCell>
                    <TableCell align="right">
                      {!REQUIRED_SECRETS.includes(name) && (
                        <Tooltip title="Delete">
                          <IconButton
                            onClick={() => handleDeleteSecret(name)}
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

      {/* Add New Secret Dialog */}
      <Dialog
        open={openNewSecretDialog}
        onClose={() => {
          setOpenNewSecretDialog(false);
          setNewSecretName('');
          setNewSecretValue('');
        }}
      >
        <DialogTitle>Add New Secret</DialogTitle>
        <DialogContent>
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
          <TextField
            margin="dense"
            label="Secret Value"
            type="password"
            fullWidth
            value={newSecretValue}
            onChange={(e) => setNewSecretValue(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setOpenNewSecretDialog(false);
            setNewSecretName('');
            setNewSecretValue('');
          }}>
            Cancel
          </Button>
          <Button
            onClick={handleAddNewSecret}
            variant="contained"
            color="primary"
            disabled={!newSecretName || !newSecretValue}
          >
            Add Secret
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}
