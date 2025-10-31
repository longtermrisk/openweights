import { createRoot } from 'react-dom/client'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import './index.css'
import App from './App.tsx'

const theme = createTheme({
  typography: {
    fontSize: 13,
    h1: { fontSize: '1.8rem' },
    h2: { fontSize: '1.6rem' },
    h3: { fontSize: '1.4rem' },
    h4: { fontSize: '1.2rem' },
    h5: { fontSize: '1.1rem' },
    h6: { fontSize: '0.95rem' },
    body1: { fontSize: '0.9rem' },
    body2: { fontSize: '0.85rem' },
    button: { fontSize: '0.85rem' },
  },
  components: {
    MuiToolbar: {
      styleOverrides: {
        root: {
          minHeight: '48px !important',
          paddingTop: '4px',
          paddingBottom: '4px',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          padding: '4px 10px',
          textTransform: 'none',
          minHeight: '32px',
        },
        sizeSmall: {
          padding: '2px 8px',
          minHeight: '28px',
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          padding: '6px',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          padding: '8px',
        },
      },
    },
    MuiCardContent: {
      styleOverrides: {
        root: {
          padding: '8px',
          '&:last-child': {
            paddingBottom: '8px',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          padding: '0px',
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          padding: '6px',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          height: '22px',
          fontSize: '0.75rem',
        },
        sizeSmall: {
          height: '20px',
          fontSize: '0.7rem',
        },
      },
    },
  },
});

createRoot(document.getElementById('root')!).render(
  <ThemeProvider theme={theme}>
    <App />
  </ThemeProvider>
)
