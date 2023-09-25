import { Card, Box, Paper, styled } from '@mui/material';

export const CardPaneItem = styled(Paper)(({ theme }) => ({
  // backgroundColor: 'linear-gradient(rgba(255, 255, 255, 0.09), rgba(255, 255, 255, 0.09))',
  backgroundColor: '#282c34',
  // ...theme.typography.body2,
  padding: '8px',
  textAlign: 'center',
  // color: theme.palette.text.primary,
  // height: '100%',
}));

export default function CardPane({ children }) {
  return (
    <Box
      display='flex'
      alignItems='center'
      justifyContent='center'
      component='div'
      sx={{
        width: '100%',
        margin: 'auto',
        flexDirection: 'column',
        height: 'auto',
        marginTop: '8px',
        marginBottom: '8px',
        // height: 'calc(100vh - 64px)',
        // '& .MuiTextField-root': { m: 2, width: '25ch' },
      }}
    >
      <Card
        raised
        sx={{
          position: 'relative',
          width: '92.5%',
          height: '90%',
          textAlign: 'center',
        }}
      >
        {children}
      </Card>
    </Box>
  );
}
