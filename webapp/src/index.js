import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { ThemeProvider } from '@mui/material';

import { createTheme } from '@mui/material';

export const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#58060E',
    },
    text: {
      primary: '#4682B4', // steelblue
    },
    action: {
      selectedOpacity: .5
    },
    background: {
      default: '#000',
    },
  },
    components: {
      MuiToggleButton: {
        styleOverrides: {
          root: {
            "&.Mui-selected": {
              color: "#000000",
              background: 'rgba(255, 165, 0, .6)'
              // backgroundColor: '#FFA500',
              // selectedOpacity: .5
            },
          }
        }
      }
    }
});


const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <ThemeProvider theme={theme}>
        <App />
    </ThemeProvider>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
