import React from "react";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faMoon, faSun, faClock, faEye, faBullseye, faStopwatch, faBed, faSeedling, faTree } from '@fortawesome/free-solid-svg-icons'
import { eventsWithinRange } from './App';

function formatMinutes(n) {
    const num = n;
    const hours = (num / 60);
    const rhours = Math.floor(hours);
    const minutes = (hours - rhours) * 60;
    let rminutes = Math.round(minutes);

    if(rminutes < 10) {
        rminutes = '0' + rminutes;
    }

    return Math.abs(rhours) + ":" + rminutes;
}
  
function formatDate(date) {
    let hours = date.getHours();
    let minutes = date.getMinutes();
    const ampm = hours >= 12 ? 'pm' : 'am';
    hours = hours % 12;
    hours = hours ? hours : 12;
    minutes = minutes < 10 ? '0' + minutes : minutes;
    const strTime = hours + ':' + minutes + ' ' + ampm;

    return strTime;
}

const getWakeWindowHours = () => {
    // TODO: map baby age to window, make a dictionary from online resources
    return 3;
}

export default function SleepStats({ sleepLogs }) {
    if(!sleepLogs) return <React.Fragment></React.Fragment>;

    const lastEvent = sleepLogs[sleepLogs.length-1];
    const twoEventsAgo = sleepLogs[sleepLogs.length-2];
  
    let displayString = lastEvent.awake === '0' ? "Asleep\n" : "Woke up\n";
    let eventTime = formatDate(lastEvent.time);
  
    const elementsToRender = [(
        <div>
          <FontAwesomeIcon style={{marginRight: '15px'}} size="3x" icon={lastEvent.awake === '0' ? faMoon : faSun} />
          <h3 style={{display: "inline-block"}}>{eventTime}<br></br>{displayString}</h3>
        </div>
      )
    ];
  
    const duration = (new Date().getTime() - new Date(lastEvent.time).getTime()) / 60000;
    
    if (lastEvent.awake === '0') { // is asleep
      const durationOfNap = Math.round((((new Date()).getTime() - new Date(lastEvent.time).getTime()) / 1000) / 60);
      const napDurationString = formatMinutes(durationOfNap);
  
      elementsToRender.push((
        <div>
          <FontAwesomeIcon style={{marginRight: '15px'}} size="3x" icon={faClock} />
          <h3 style={{display: "inline-block"}}>{napDurationString}<br></br>Slept</h3>
        </div>
      ));
    } else { // is awake
      const durationOfNap = Math.round((((lastEvent.time).getTime() - new Date(twoEventsAgo.time).getTime()) / 1000) / 60);
      const napDurationString = formatMinutes(durationOfNap);
  
      elementsToRender.push((
        <div>
          <FontAwesomeIcon style={{marginRight: '15px'}} size="3x" icon={faClock} />
          <h3 style={{display: "inline-block", marginRight: '45px'}}>{napDurationString}<br></br>Slept</h3>
        </div>
      ));
  
      elementsToRender.push((
        <div>
          <FontAwesomeIcon style={{marginRight: '15px'}} size="3x" icon={faEye} />
          <h3 style={{display: "inline-block", marginRight: '20px'}}>{formatMinutes(Math.round(duration))}<br></br>Awake</h3>
        </div>
      ));
  
      const windowHours = getWakeWindowHours();
      const targetSleepTime = new Date(new Date(lastEvent.time).getTime() + (windowHours * 60 * 60 * 1000)) // N hours from when woke up
  
      const minutesFromWindowWarning = 15;
      const warningThreshold = new Date(targetSleepTime.getTime() - minutesFromWindowWarning * 60000);
      const isWarning = new Date() > warningThreshold;
      const isPastWindow = new Date() > targetSleepTime;
      const warningColor = isPastWindow ? '#ff0000' : isWarning ? '#FFFF00' : 'inherit';
      elementsToRender.push((
        <div style={{color: warningColor}}>
          <FontAwesomeIcon style={{marginRight: '15px'}} size="3x" icon={faBullseye} />
          <h3 style={{display: "inline-block", marginRight: "20px"}}>{formatDate(targetSleepTime)}<br></br>Window</h3>
        </div>
      ));
    }
  
    // Count nap(s) duration "today", between 7am - 7pm
    const beginningOfNapWindow = new Date();
    beginningOfNapWindow.setHours(7,0,0,0);
    const endOfNapWindow = new Date();
    endOfNapWindow.setHours(19,0,0,0);
  
    const logsToAggregate = eventsWithinRange(sleepLogs, beginningOfNapWindow, endOfNapWindow);
  
    // TODO: Clean up edge cases, naps/night sleep overlapping "day" hours
  
    const napDurationList = logsToAggregate.map((event, i) => {
      if(event.awake == 0 && i+1 < logsToAggregate.length) {
        const duration = logsToAggregate[i+1].time - logsToAggregate[i].time;
        const diffMins = Math.round((duration / 1000) / 60); // minutes
        return diffMins;
      }
      return 0;
    });
  
    const sumMinutes = napDurationList.reduce((partialSum, a) => partialSum + a, 0);
    elementsToRender.push((
      <div>
        <FontAwesomeIcon style={{marginRight: '15px'}} size="3x" icon={faStopwatch} />
        <h3 style={{display: "inline-block"}}>{formatMinutes(sumMinutes)}<br></br>Total nap</h3>
      </div>
    ));
  
  
    // Get total night sleep between 7pm - 7am
    const startOfLastNight = new Date()
    startOfLastNight.setDate(startOfLastNight.getDate() - 1);
    startOfLastNight.setHours(18,45,0,0);
    const endOfLastNight = new Date();
    endOfLastNight.setHours(8,0,0,0);
  
    const logsToAggregate1 = eventsWithinRange(sleepLogs, startOfLastNight, endOfLastNight);
  
    // TODO: Clean up edge cases, naps/night sleep overlapping "day" hours
  
    const lastNightSleepDurationList = logsToAggregate1.map((event, i) => {
      if(event.awake == 0 && i+1 < logsToAggregate1.length) {
        const duration = logsToAggregate1[i+1].time - logsToAggregate1[i].time;
        const diffMins = Math.round((duration / 1000) / 60); // minutes
        return diffMins;
      }
      return 0;
    });
  
    const sumMinutesSleepLastNight = lastNightSleepDurationList.reduce((partialSum, a) => partialSum + a, 0);
    elementsToRender.push((
      <div>
        <FontAwesomeIcon style={{marginRight: '15px'}} size="3x" icon={faBed} />
        <h3 style={{display: "inline-block"}}>{formatMinutes(sumMinutesSleepLastNight)}<br></br>Last night</h3>
      </div>
    ));

    return (
      <div id="displayText" style={{ textAlign: 'center', display: 'grid', gridTemplateColumns: '50% 50%' }}>
        {
          elementsToRender.map((element, i) => <div key={i}>{element}</div>)
        }
      </div>
    );
  }
  