import { ToggleButton, ToggleButtonGroup } from '@mui/material';
import ViewColumnIcon from '@mui/icons-material/ViewColumn';
import ViewListIcon from '@mui/icons-material/ViewList';

interface ViewToggleProps {
  view: 'three-column' | 'list';
  onViewChange: (view: 'three-column' | 'list') => void;
}

export function ViewToggle({ view, onViewChange }: ViewToggleProps) {
  const handleChange = (_: React.MouseEvent<HTMLElement>, newView: 'three-column' | 'list' | null) => {
    if (newView !== null) {
      onViewChange(newView);
    }
  };

  return (
    <ToggleButtonGroup
      value={view}
      exclusive
      onChange={handleChange}
      aria-label="view mode"
      size="small"
    >
      <ToggleButton value="three-column" aria-label="three column view">
        <ViewColumnIcon />
      </ToggleButton>
      <ToggleButton value="list" aria-label="list view">
        <ViewListIcon />
      </ToggleButton>
    </ToggleButtonGroup>
  );
}
