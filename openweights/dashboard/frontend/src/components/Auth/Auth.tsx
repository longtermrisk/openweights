import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  TextField,
  Typography,
  Link,
  Container,
  Tabs,
  Tab,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import { useAuth } from '../../contexts/AuthContext';

type AuthMode = 'signin' | 'signup' | 'reset';

export function Auth() {
  const [mode, setMode] = useState<AuthMode>('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [resetDialogOpen, setResetDialogOpen] = useState(false);
  const [resetEmail, setResetEmail] = useState('');

  const { signIn, signUp, resetPassword } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    try {
      if (mode === 'signup') {
        if (password !== confirmPassword) {
          setError('Passwords do not match');
          return;
        }
        const { error } = await signUp(email, password);
        if (error) throw error;
        setSuccess('Sign up successful! Please check your email for verification.');
        // After successful signup, sign in and redirect to organizations
        const { error: signInError } = await signIn(email, password);
        if (!signInError) {
          navigate('/organizations');
        }
      } else if (mode === 'signin') {
        const { error } = await signIn(email, password);
        if (error) throw error;
        navigate('/jobs');
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'An error occurred');
    }
  };

  const handlePasswordReset = async () => {
    try {
      const { error } = await resetPassword(resetEmail);
      if (error) throw error;
      setSuccess('Password reset email sent! Please check your inbox.');
      setResetDialogOpen(false);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'An error occurred');
    }
  };

  return (
    <Container component="main" maxWidth="xs">
      <Box
        sx={{
          marginTop: 8,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <img
          src="/ow.svg"
          alt="OpenWeights Logo"
          style={{
            height: '128px',
            width: '128px',
            marginBottom: '2rem'
          }}
        />

        <Typography component="h1" variant="h5" gutterBottom>
          Welcome to OpenWeights
        </Typography>

        <Link
          href="https://github.com/longtermrisk/openweights/blob/main/README.md"
          target="_blank"
          rel="noopener noreferrer"
          sx={{ mb: 4 }}
        >
          View Documentation
        </Link>



        <Tabs
          value={mode}
          onChange={(_, newValue: AuthMode) => {
            setMode(newValue);
            setError('');
            setSuccess('');
          }}
          sx={{ mb: 3 }}
        >
          <Tab label="Sign In" value="signin" />
          <Tab label="Sign Up" value="signup" />
        </Tabs>

        {error && (
        <Alert severity="error" sx={{ mb: 2, width: '100%' }}>
          {error}
        </Alert>
      )}

      {success && (
        <>
          <Alert severity="success" sx={{ mb: 2, width: '100%' }}>
            {success}
          </Alert>
          <Alert severity="info" sx={{ mb: 2, width: '100%' }}>
            Note: Our emails may be delivered to your spam folder. Please check there if you don't see them in your inbox.
          </Alert>
        </>
      )}

        <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1, width: '100%' }}>
          <TextField
            margin="normal"
            required
            fullWidth
            id="email"
            label="Email Address"
            name="email"
            autoComplete="email"
            autoFocus
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />

          <TextField
            margin="normal"
            required
            fullWidth
            name="password"
            label="Password"
            type="password"
            id="password"
            autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />

          {mode === 'signup' && (
            <TextField
              margin="normal"
              required
              fullWidth
              name="confirmPassword"
              label="Confirm Password"
              type="password"
              id="confirmPassword"
              autoComplete="new-password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
            />
          )}

          <Button
            type="submit"
            fullWidth
            variant="contained"
            sx={{ mt: 3, mb: 2 }}
          >
            {mode === 'signin' ? 'Sign In' : 'Sign Up'}
          </Button>

          {mode === 'signin' && (
            <Box sx={{ textAlign: 'center' }}>
              <Link
                component="button"
                variant="body2"
                onClick={(e) => {
                  e.preventDefault();
                  setResetDialogOpen(true);
                }}
              >
                Forgot password?
              </Link>
            </Box>
          )}
        </Box>
      </Box>

      {/* Password Reset Dialog */}
      <Dialog open={resetDialogOpen} onClose={() => setResetDialogOpen(false)}>
        <DialogTitle>Reset Password</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            id="resetEmail"
            label="Email Address"
            type="email"
            fullWidth
            variant="outlined"
            value={resetEmail}
            onChange={(e) => setResetEmail(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResetDialogOpen(false)}>Cancel</Button>
          <Button onClick={handlePasswordReset}>Send Reset Link</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}
