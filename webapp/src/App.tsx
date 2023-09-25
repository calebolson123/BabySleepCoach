// @ts-nocheck
import React, { useEffect, useState } from "react";
import * as d3 from "d3";
import Retrain from "./Retrain";
import CardPane from './card';
import VideoFeed, { VideoFeedTypeEnum } from './VideoFeed';
import SleepStats from './SleepStats';
import Charts from './Charts';
import Settings from './Settings';

export function eventsWithinRange(events: any, startDate: any, endDate: any) {
  return events.filter((log: any) => {
    const logTime = log.time;
    return logTime > startDate && logTime < endDate;
  });
}

const getSleepLogs = async (forecast: any) => {

  const file = forecast ? 'sleep_logs_forecasted' : 'sleep_logs';
  // Request sleep log data from HTTP server on the device (somewhere on LAN) which is running with sleep tracking service
  const sleepLogs = await d3.csv(`http://${process.env.REACT_APP_RESOURCE_SERVER_IP}/${file}.csv`);

  // convert timestamps to date objects
  sleepLogs.forEach((d) => {
    d.time = new Date(d.time * 1000);
  });

  return sleepLogs;
}

function App() {

  const [sleepLogs, setSleepLogs] = useState(null);
  const [forecast, setForecast] = useState(false);
  const [modelProba, setModelProba] = useState<any>(0.5);
  console.log('modelProba: ', modelProba);
  const [videoFeedType, setVideoFeedType] = useState<VideoFeedTypeEnum>(VideoFeedTypeEnum.RAW)

  useEffect(() => {
    getSleepLogs(forecast).then(sleepLogs => {
      setSleepLogs(sleepLogs);
    });
  }, [forecast]);

  return (
    <div>
      <CardPane>
        <h2 style={{ color: 'orange' }}>The Baby Sleep Coach</h2>
      </CardPane>
      <CardPane>
        <SleepStats sleepLogs={sleepLogs} />
      </CardPane>
      <CardPane>
        <Retrain videoFeedType={videoFeedType} />
        <VideoFeed modelProba={modelProba} setModelProba={setModelProba} videoFeedType={videoFeedType} setVideoFeedType={setVideoFeedType} />
      </CardPane>

      <Charts sleepLogs={sleepLogs} forecast={forecast} setForecast={setForecast} />

      <CardPane>
        <Settings />
      </CardPane>
    </div>
  );
}

export default App