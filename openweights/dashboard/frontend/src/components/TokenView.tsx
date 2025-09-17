import { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  IconButton,
  Snackbar,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent
} from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DeleteIcon from '@mui/icons-material/Delete';
import { useAuth } from '../contexts/AuthContext';
import { api } from '../api';

interface Token {
  id: string;
  name: string;
  expires_at: string | null;
  created_at: string;
  access_token?: string;
}

interface TokenViewProps {
  orgId: string;
}

export function TokenView({ orgId }: TokenViewProps) {
  const { user } = useAuth();
  const [showSnackbar, setShowSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [tokens, setTokens] = useState<Token[]>([]);
  const [openDialog, setOpenDialog] = useState(false);
  const [newTokenName, setNewTokenName] = useState('');
  const [expirationDays, setExpirationDays] = useState<string>('never');
  const [newToken, setNewToken] = useState<Token | null>(null);

  // Fetch tokens
  useEffect(() => {
    const fetchTokens = async () => {
      try {
        const tokens = await api.listTokens(orgId);
        setTokens(tokens);
      } catch (error) {
        console.error('Error fetching tokens:', error);
        setSnackbarMessage('Error fetching tokens');
        setShowSnackbar(true);
      }
    };

    fetchTokens();
  }, [orgId]);

  const handleCopy = async (text: string) => {
    await navigator.clipboard.writeText(text);
    setSnackbarMessage('Token copied to clipboard');
    setShowSnackbar(true);
  };

  const handleExpirationChange = (event: SelectChangeEvent) => {
    setExpirationDays(event.target.value);
  };

  const handleCreateToken = async () => {
    if (!newTokenName.trim()) {
      setSnackbarMessage('Please enter a token name');
      setShowSnackbar(true);
      return;
    }

    try {
      const token = await api.createToken(
        orgId,
        newTokenName.trim(),
        expirationDays === 'never' ? undefined : parseInt(expirationDays)
      );
      setNewToken(token);

      // Refresh token list
      const updatedTokens = await api.listTokens(orgId);
      setTokens(updatedTokens);

      setOpenDialog(false);
      setNewTokenName('');
      setExpirationDays('never');
    } catch (error) {
      setSnackbarMessage(error instanceof Error ? error.message : 'Error creating token');
      setShowSnackbar(true);
    }
  };

  const handleDeleteToken = async (tokenId: string) => {
    try {
      await api.deleteToken(orgId, tokenId);

      // Refresh token list
      const tokens = await api.listTokens(orgId);
      setTokens(tokens);

      setSnackbarMessage('Token deleted successfully');
      setShowSnackbar(true);
    } catch (error) {
      setSnackbarMessage(error instanceof Error ? error.message : 'Error deleting token');
      setShowSnackbar(true);
    }
  };

  if (!user) {
    return (
      <Typography variant="body1" color="error" sx={{ p: 3 }}>
        You need to be logged in to manage tokens
      </Typography>
    );
  }

  return (
    <Box>
      <Button
        variant="contained"
        onClick={() => setOpenDialog(true)}
        sx={{ mb: 3 }}
      >
        Create New Token
      </Button>

      {newToken && (
        <Paper sx={{ p: 2, mb: 3, bgcolor: 'success.light' }}>
          <Typography variant="subtitle1" gutterBottom>
            New token created! Make sure to copy it now - you won't be able to see it again.
          </Typography>
          <Box sx={{
            display: 'flex',
            alignItems: 'center',
            bgcolor: 'background.paper',
            p: 2,
            borderRadius: 1
          }}>
            <Typography
              variant="body2"
              sx={{
                fontFamily: 'monospace',
                flexGrow: 1,
                overflow: 'hidden',
                textOverflow: 'ellipsis'
              }}
            >
              {newToken.access_token}
            </Typography>
            <IconButton onClick={() => handleCopy(newToken.access_token || '')} size="small">
              <ContentCopyIcon />
            </IconButton>
          </Box>
        </Paper>
      )}

      <List>
        {tokens.map((token) => (
          <ListItem
            key={token.id}
            sx={{
              bgcolor: 'background.paper',
              mb: 1,
              borderRadius: 1
            }}
          >
            <ListItemText
              primary={token.name}
              secondary={
                <>
                  Created: {new Date(token.created_at).toLocaleString()}
                  {token.expires_at && (
                    <><br />Expires: {new Date(token.expires_at).toLocaleString()}</>
                  )}
                </>
              }
            />
            <ListItemSecondaryAction>
              <IconButton
                edge="end"
                aria-label="delete"
                onClick={() => handleDeleteToken(token.id)}
              >
                <DeleteIcon />
              </IconButton>
            </ListItemSecondaryAction>
          </ListItem>
        ))}
      </List>

      <Dialog open={openDialog} onClose={() => setOpenDialog(false)}>
        <DialogTitle>Create New API Token</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Token Name"
            fullWidth
            variant="outlined"
            value={newTokenName}
            onChange={(e) => setNewTokenName(e.target.value)}
            sx={{ mb: 2 }}
          />
          <FormControl fullWidth>
            <InputLabel>Token Expiration</InputLabel>
            <Select
              value={expirationDays}
              label="Token Expiration"
              onChange={handleExpirationChange}
            >
              <MenuItem value="never">Never</MenuItem>
              <MenuItem value="30">30 days</MenuItem>
              <MenuItem value="90">90 days</MenuItem>
              <MenuItem value="180">180 days</MenuItem>
              <MenuItem value="365">1 year</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
          <Button onClick={handleCreateToken} variant="contained">Create</Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={showSnackbar}
        autoHideDuration={2000}
        onClose={() => setShowSnackbar(false)}
        message={snackbarMessage}
      />
    </Box>
  );
}
