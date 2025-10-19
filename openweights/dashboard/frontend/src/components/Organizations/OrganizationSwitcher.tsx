import { Select, MenuItem, FormControl, SelectChangeEvent, Box, Divider, ListItemIcon, ListItemText } from '@mui/material';
import { useOrganization } from '../../contexts/OrganizationContext';
import { useNavigate } from 'react-router-dom';
import AddIcon from '@mui/icons-material/Add';

export function OrganizationSwitcher() {
  const { currentOrganization, setCurrentOrganization, organizations } = useOrganization();
  const navigate = useNavigate();

  const handleChange = (event: SelectChangeEvent<string>) => {
    const selectedValue = event.target.value;

    // Check if user clicked "Create New Organization"
    if (selectedValue === '__create_new__') {
      navigate('/organizations');
      return;
    }

    const org = organizations.find(o => o.id === selectedValue);
    if (org) {
      setCurrentOrganization(org);
    }
  };

  if (!currentOrganization || organizations.length === 0) {
    return null;
  }

  return (
    <Box sx={{ mx: 2 }}>
      <FormControl size="small" sx={{ minWidth: 200 }}>
        <Select
          value={currentOrganization.id}
          onChange={handleChange}
          sx={{
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            color: 'white',
            '& .MuiSelect-icon': { color: 'white' },
            '&:before': { borderColor: 'rgba(255, 255, 255, 0.3)' },
            '&:hover:not(.Mui-disabled):before': { borderColor: 'rgba(255, 255, 255, 0.5)' },
          }}
        >
          {organizations.map((org) => (
            <MenuItem key={org.id} value={org.id}>
              {org.name}
            </MenuItem>
          ))}
          <Divider />
          <MenuItem value="__create_new__">
            <ListItemIcon sx={{ minWidth: 32 }}>
              <AddIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Create New Organization" />
          </MenuItem>
        </Select>
      </FormControl>
    </Box>
  );
}
