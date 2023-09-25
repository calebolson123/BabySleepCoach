import React, { useState } from 'react';
import { Grid, Box, Checkbox } from '@mui/material';

export default function Settings() {

    const [checked, setChecked] = useState<boolean | undefined>(undefined)

    const getSleepNotificationsEnabled = async () => {
        const enabled = await ((await fetch(`http://${process.env.REACT_APP_BACKEND_IP}/getSleepNotificationsEnabled`)).text());
        console.log('result: ', enabled);
        setChecked(enabled === 'true');
    };

    const sleepNotificationEnabled = getSleepNotificationsEnabled();
    console.log('sleepNotificationEnabled: ', sleepNotificationEnabled);

    const handleChange = async (_e: any, enabled: boolean) => {
        console.log('enabled: ', enabled);
        await fetch(`http://${process.env.REACT_APP_BACKEND_IP}/setSleepNotificationsEnabled/${enabled}`);
        console.log('done.');
        getSleepNotificationsEnabled();
    }

    return (
        <Grid container>
            <Grid item xs={12}>
                <h4>Settings</h4>
            </Grid>
            <Grid item xs={12}>
                <Box component='div' display='flex' flexDirection="row" justifyContent='start' alignItems='center'>
                    <Checkbox onChange={handleChange} disabled={checked == undefined} checked={checked !== undefined ? checked : false} sx={{ color: 'orange !important' }} /><span style={{ marginLeft: '-5px' }}>Wake notifications</span>
                </Box>
            </Grid>
        </Grid>
    );
}
