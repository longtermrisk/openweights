import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Box,
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from '@mui/material'
import { supabase } from '../../supabaseClient'

interface Organization {
  id: string
  name: string
  role: 'admin' | 'user'
}

export function OrganizationsList() {
  const [organizations, setOrganizations] = useState<Organization[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [openCreateDialog, setOpenCreateDialog] = useState(false)
  const [newOrgName, setNewOrgName] = useState('')

  useEffect(() => {
    fetchOrganizations()
  }, [])

  async function fetchOrganizations() {
    try {
      const { data, error: membershipError } = await supabase
        .from('organization_members')
        .select(`
          organization_id,
          role,
          organization:organization_id (
            id,
            name
          )
        `)
        .throwOnError()

      if (membershipError) throw membershipError

      // Transform and deduplicate organizations
      const orgMap = new Map<string, Organization>()
      if (data) {
        data.forEach((membership: any) => {
          if (membership.organization) {
            const org: Organization = {
              id: membership.organization.id,
              name: membership.organization.name,
              role: membership.role
            }
            // If org already exists, only update if new role is admin
            if (!orgMap.has(org.id) || org.role === 'admin') {
              orgMap.set(org.id, org)
            }
          }
        })
      }

      setOrganizations(Array.from(orgMap.values()))
    } catch (err) {
      console.error('Error fetching organizations:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch organizations')
    } finally {
      setLoading(false)
    }
  }

  const handleCreateOrganization = async () => {
    try {
      const { error } = await supabase
        .rpc('create_organization', {
          org_name: newOrgName
        })

      if (error) throw error

      // Refresh organizations list
      await fetchOrganizations()
      setOpenCreateDialog(false)
      setNewOrgName('')
    } catch (err) {
      console.error('Error creating organization:', err)
      setError(err instanceof Error ? err.message : 'Failed to create organization')
    }
  }

  if (loading) return <Typography>Loading...</Typography>
  if (error) return <Typography color="error">{error}</Typography>

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">
          Your Organizations
        </Typography>
        <Button
          variant="contained"
          color="primary"
          onClick={() => setOpenCreateDialog(true)}
        >
          Create Organization
        </Button>
      </Box>

      <List>
        {organizations.map((org) => (
          <ListItem key={org.id} component={Card} sx={{ mb: 2 }}>
            <CardContent sx={{ width: '100%' }}>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Box>
                  <Typography variant="h6" component={Link} to={`/organizations/${org.id}`} sx={{ textDecoration: 'none' }}>
                    {org.name}
                  </Typography>
                  <Typography color="textSecondary" variant="body2">
                    ID: {org.id}
                  </Typography>
                </Box>
                <Chip
                  label={org.role}
                  color={org.role === 'admin' ? 'primary' : 'default'}
                  variant={org.role === 'admin' ? 'filled' : 'outlined'}
                />
              </Box>
            </CardContent>
          </ListItem>
        ))}
      </List>

      <Dialog open={openCreateDialog} onClose={() => setOpenCreateDialog(false)}>
        <DialogTitle>Create New Organization</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Organization Name"
            type="text"
            fullWidth
            value={newOrgName}
            onChange={(e) => setNewOrgName(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenCreateDialog(false)}>Cancel</Button>
          <Button
            onClick={handleCreateOrganization}
            variant="contained"
            color="primary"
            disabled={!newOrgName.trim()}
          >
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}
