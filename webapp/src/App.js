

import React, { useEffect, useState } from "react";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faMoon, faSun, faClock, faEye, faBullseye, faStopwatch, faBed, faSeedling, faTree } from '@fortawesome/free-solid-svg-icons'
import * as d3 from "d3";

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

function eventsWithinRange(events, startDate, endDate) {
  return events.filter(log => {
    const logTime = log.time;
    return logTime > startDate && logTime < endDate;
  });
}

// Generate and write SVG chart
// TODO: make more responsive & less hardcoded
const createChart = (data, rangeStart, rangeEnd) => {
  const margin = { top: 20, right: 20, bottom: 50, left: 70 },
  width = 2000 - margin.left - margin.right,
  height = 120 - margin.top - margin.bottom;

  // append the svg object to the body of the page
  const svg = d3.select("#graph").append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .style("fill", "steelblue")
      .append("g")
      .attr("transform", `translate(${margin.left}, ${margin.top})`)
      .attr("stroke", "steelblue");
      
  svg.append("text")
    .attr("x", 100)             
    .attr("y", 0 - (margin.top / 2))
    .attr("text-anchor", "start")  
    .style("font-size", "13px") 
    .attr("stroke-width", 1)
    .text("Today's Sleep");

  const x = d3.scaleTime()
    .range([0, width])
    .domain([rangeStart, rangeEnd]);

  const y = d3.scaleLinear().range([height, 0]);

  svg.append("g")
    .attr("transform", `translate(0, ${height})`)
    .call(d3.axisBottom(x).ticks(24))
    .attr("stroke", "steelblue");

  svg.append("g")
    .call(d3.axisLeft(y).ticks(1, "d").tickFormat((d, i) => i === 0 ? "awake" : "asleep"))
    .attr("stroke-width", .75)
    .attr("stroke", "steelblue");
    
  const valueLine = d3.line().curve(d3.curveStepAfter)
    .x((d) => { return x(d.time); })
    .y((d) => { return y(d.awake === '1' ? '0' : '1'); }) // invert values for graphing purposes

  svg.append("path")
    .data([data])
    .attr("class", "line")
    .attr("fill", "none")
    .attr("stroke", "steelblue")
    .attr("stroke-width", 1.5)
    .attr("d", valueLine);
}

const drawCircleSector = (ctx, radius, startAngleDeg, endAngleDeg, color) => {
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.strokeStyle = color;

  // Stop at 360 and start drawing at next radius increment (every 24 hours)
  if(startAngleDeg > endAngleDeg) { // i.e. end of this rev is past 12am
    ctx.arc(window.innerWidth/2, window.innerHeight/3 + 50, radius, dToR(startAngleDeg-90), dToR(360.5-90));
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(window.innerWidth/2, window.innerHeight/3 + 50, radius + 2, dToR(0-90), dToR(endAngleDeg-90));
  } else {
    ctx.arc(window.innerWidth/2, window.innerHeight/3 + 50, radius, dToR(startAngleDeg-90), dToR(endAngleDeg-90));
  }

  ctx.stroke();
}

function dToR(degrees) {
  var pi = Math.PI;
  return degrees * (pi/180);
}

// Given a date, returns the % of the 24 hour day elapsed.
// This is used for drawing each sector of the
// crazy concentric circle chart
const percentageOfDayElapsed = (d) => {
  return (d.getHours() * 3600 + d.getMinutes() * 60 + d.getSeconds() + d.getMilliseconds()/1000)/86400;
}

// draw a circle of wake & sleep sectors for each 24 hour day
const createCrazyCircles = (events, forecast) => {
  const canvas = document.getElementById("myCanvas");
  const context = canvas.getContext("2d");

  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight-200;

  const initialRadius = 30
  const offset = 50;

  const img = new Image();
  img.src = "http://this_device_ip:8000//absolute_path_to_project_dir/webapp/public/baby_face_192.png";
  img.addEventListener("load", (e) => {
      context.save();
      context.beginPath();
      context.arc(window.innerWidth/2, window.innerHeight/3 + offset, 50, 0, Math.PI * 2, true);
      context.closePath();
      context.clip();

      context.drawImage(img, window.innerWidth/2 - 50, window.innerHeight/3 - 50 + offset, 100, 100);

      context.beginPath();
      context.arc(window.innerWidth/2 - 50, window.innerHeight/3 - 50 + offset, 50, 0, Math.PI * 2, true);
      context.clip();
      context.closePath();
      context.restore();
  });
  
  let dayCount = 0;
  const firstDay = events[0].time;

  let endA;
  // let dayDiff = 0;
  let revCounter = 0;

  const month = firstDay.getMonth() + 1;
  const day = firstDay.getDate();
  const year = firstDay.getFullYear();
  const uniqueDates = new Set();
  uniqueDates.add(day.toString() + month.toString() + year.toString())
  events.forEach((e, i) => {
      function bruh() { // artificial delay for growing circles animation
        setTimeout(() => {
          if(i+1 < events.length) {
            const timeDiff = e.time.getTime() - firstDay.getTime();
            // dayDiff = Math.round(timeDiff / (1000 * 3600 * 24));

            const thisMonth = e.time.getMonth() + 1;
            const thisDay = e.time.getDate();
            const thisYear = e.time.getFullYear();
            const thisDate = thisDay.toString() + thisMonth.toString() + thisYear.toString();
            if(!uniqueDates.has(thisDate)) {
              revCounter = revCounter + 1;
            }
            uniqueDates.add(thisDate)
        
            const tStart = events[i].time;
            const tEnd = events[i+1].time;
            const percentOfDayElapsedStart = percentageOfDayElapsed(tStart)
            const percentOfDayElapsedEnd = percentageOfDayElapsed(tEnd)

            const startA = percentOfDayElapsedStart * 360;
            endA = percentOfDayElapsedEnd * 360;

            const color = e.awake === '0' ? 'blue' : 'orange';
            drawCircleSector(context, initialRadius + (revCounter*2) + 25, startA, endA, color);
          }
        }, 1000)
    }

    requestAnimationFrame(bruh);
  });

  setTimeout(() => {
    if (!forecast) {
      // final marker where circle is growing from
      drawCircleSector(context, initialRadius + (revCounter*2) + 25, endA, endA + 5, '#66FF00');
    }

    // clock marks
    context.fillStyle = "steelblue";
    context.font = "bold italic 12px sans-serif";
    context.fillText("12 AM", window.innerWidth/2 - 15, window.innerHeight/3 - 100 + offset - (revCounter*1.5));
    context.fillText("12 PM", window.innerWidth/2 - 15, window.innerHeight/3 + 110 + offset + (revCounter*1.5));
    
    // title
    context.font = "bold 26px sans-serif";
    const titleText = forecast  ? "Sleep Forecast" : "Sleepy Sprout"
    context.fillText(titleText, window.innerWidth/2 - 80, window.innerHeight/3 - 110 - revCounter);

    // subtitle
    const subtitleText = forecast  ? "Next month forecasted" : "Every day a ring is added"
    context.font = "bold italic 12px sans-serif";
    context.fillText(subtitleText, window.innerWidth/2 - 60, window.innerHeight/3 - 95 - revCounter);

    // legend
    context.fillStyle = "blue";
    context.fillRect(window.innerWidth/2 - 105, window.innerHeight/3 + 100 + offset + revCounter*2, 15, 15)

    context.fillStyle = "orange";
    context.fillRect(window.innerWidth/2 - 105, window.innerHeight/3 + 125 + offset + revCounter*2, 15, 15)

    context.fillStyle = "steelblue";
    context.font = "14px sans-serif";
    context.fillText("Awake", window.innerWidth/2 - 80, window.innerHeight/3 + 137 + offset + revCounter*2);
    context.fillText("Asleep", window.innerWidth/2 - 80, window.innerHeight/3 + 113 + offset + revCounter*2);
  }, forecast ? 3500 : 2500);
}

const getWakeWindowHours = () => {
  // TODO: map baby age to window, make a dictionary from online resources
  return 2.5;
}


// TODO: if care, make this more react-y, components
const buildJSX = (sleep_logs) => {

  const lastEvent = sleep_logs[sleep_logs.length-1];
  const twoEventsAgo = sleep_logs[sleep_logs.length-2];

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
        <h3 style={{display: "inline-block"}}>{napDurationString}<br></br>Slept</h3>
      </div>
    ));

    elementsToRender.push((
      <div>
        <FontAwesomeIcon style={{marginRight: '15px'}} size="3x" icon={faEye} />
        <h3 style={{display: "inline-block"}}>{formatMinutes(Math.round(duration))}<br></br>Awake</h3>
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
        <h3 style={{display: "inline-block"}}>{formatDate(targetSleepTime)}<br></br>Window</h3>
      </div>
    ));
  }

  // Count nap(s) duration "today", between 7am - 7pm
  const beginningOfNapWindow = new Date();
  beginningOfNapWindow.setHours(7,0,0,0);
  const endOfNapWindow = new Date();
  endOfNapWindow.setHours(19,0,0,0);

  const logsToAggregate = eventsWithinRange(sleep_logs, beginningOfNapWindow, endOfNapWindow);

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

  const logsToAggregate1 = eventsWithinRange(sleep_logs, startOfLastNight, endOfLastNight);

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

  return elementsToRender;
}

Date.prototype.addDays = function(days) {
  var date = new Date(this.valueOf());
  date.setDate(date.getDate() + days);
  return date;
}

// TODO: break this up
// This function gets sleep data and builds JSX to render charts and stats.
const processSleepLogs = async (forecast) => {

  const file = forecast ? 'sleep_logs_forecasted' : 'sleep_logs';
  // Request sleep log data from HTTP server on the device (somewhere on LAN) which is running with sleep tracking service
  const sleepLogs = await d3.csv(`http://ip_address_of_device_running_sleep_tracker:8000/path_on_device_to_sleep_log_csv_file/${file}.csv`);

  // convert timestamps to date objects
  sleepLogs.forEach((d) => {
    d.time = new Date(d.time * 1000);
  });

  const beginningOfChartRange= new Date().addDays(-1);
  beginningOfChartRange.setHours(19,0,0,0);
  const endOfChartRange = new Date().addDays(1);
  endOfChartRange.setHours(6,0,0,0);

  const chartEvents = eventsWithinRange(sleepLogs, beginningOfChartRange, endOfChartRange);

  const lastEvent = sleepLogs[sleepLogs.length-1];
  const twoEventsAgo = sleepLogs[sleepLogs.length-2];
  const toRender = buildJSX(sleepLogs);

  // create and insert chart in body
  createChart(chartEvents, beginningOfChartRange, endOfChartRange);
  createCrazyCircles(sleepLogs, forecast);

  return toRender;
}

// TODO: break apart this file into components
function App() {

  const [info, setInfo] = useState(null);
  const [forecast, setForecast] = useState(false);

  useEffect(() => {
    processSleepLogs(forecast).then(result => {
      setInfo(result);
    })
  }, [forecast]);

  return (
    // TODO: don't inline styles
    <div>
      <div id="displayText" style={{marginTop: '50px', textAlign: 'center', display: 'grid', gridTemplateColumns: '50% 50%'}}>
        {info}
      </div>
      <div id="chartContainer" style={{ width: '100vw', height: '100%', overflow: 'auto', marginTop: '75px'}}>
        <div style={{ width: '100%', height: '100%'}}>
          <div id="graph"></div>
        </div>
      </div>
      <div>
        <canvas style={{marginTop: "30px"}} id="myCanvas"></canvas>
      </div>
      <div style={{textAlign: 'center', marginBottom: '150px'}} onClick={() => setForecast(!forecast)}>
        <div style={{marginBottom: '100px'}}>
          <div style={{color: 'blue', borderRadius: '15px', textAlign: 'center', margin: 'auto', width: '70px', boxShadow: '0 0 30px 3px rgba(255,165,0,0.7)', backgroundColor: 'rgba(255,165,0,0.75)' }}>
            <FontAwesomeIcon size="3x" icon={forecast ? faTree : faSeedling} />
          </div>
          <span style={{color: 'steelblue', textAlign: 'center', margin: 'auto', marginBottom: '100px'}}>{forecast ? 'Historical' : 'Forecast'}</span>
        </div>
      </div>
    </div>
  );
}

export default App