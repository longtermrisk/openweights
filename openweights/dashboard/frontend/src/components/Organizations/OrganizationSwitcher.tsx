import { Select, MenuItem, FormControl, SelectChangeEvent } from '@mui/material';
import { useOrganization } from '../../contexts/OrganizationContext';

export function OrganizationSwitcher() {
  const { currentOrganization, setCurrentOrganization, organizations } = useOrganization();

  const handleChange = (event: SelectChangeEvent<string>) => {
    const org = organizations.find(o => o.id === event.target.value);
    if (org) {
      setCurrentOrganization(org);
    }
  };

  if (!currentOrganization || organizations.length === 0) {
    return null;
  }

  return (
    <FormControl size="small" sx={{ minWidth: 200, mx: 2 }}>
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
      </Select>
    </FormControl>
  );
}
