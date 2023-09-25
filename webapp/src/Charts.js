import React, { useMemo } from "react";
import eventsWithinRange from './App';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSeedling, faTree } from '@fortawesome/free-solid-svg-icons';
import * as d3 from "d3";
import CardPane from './card';

const createChart = (data, rangeStart, rangeEnd) => {
    const margin = { top: 20, right: 20, bottom: 50, left: 70 },
    width = 2000 - margin.left - margin.right,
    height = 120 - margin.top - margin.bottom;
  
    // append the svg object to the body of the page
    d3.select("#graph svg").remove();
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
    if(!canvas) return;
    const context = canvas.getContext("2d");
  
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight-200;
  
    const initialRadius = 0
    const offset = 50;

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

    d3.select(context.canvas).call(d3.zoom()
        .scaleExtent([-10, 10])
        .on("zoom", ({transform}) => zoomed(transform)));

    function zoomed(transform) {
      context.save();
      context.clearRect(0, 0, window.innerWidth, window.innerHeight);
      context.translate(transform.x, transform.y);
      context.scale(transform.k, transform.k);
      context.beginPath();

      const uniqueDates = new Set();
      revCounter = 0;
      events.forEach((e, i) => {
        // function bruh() { // artificial delay for growing circles animation
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
            // console.log('revCounter: ', revCounter)
            const color = e.awake === '0' ? 'blue' : 'orange';
            drawCircleSector(context, (initialRadius + (revCounter*3)), startA, endA, color);
          }
        // }
        // bruh()
        // requestAnimationFrame(bruh);
      });
      drawCircleSector(context, initialRadius + (revCounter*3), endA, endA + 5, '#66FF00');

      context.fill();
      context.restore();
    }

    zoomed(d3.zoomIdentity);

  }

  Date.prototype.addDays = function(days) {
    var date = new Date(this.valueOf());
    date.setDate(date.getDate() + days);
    return date;
  }

function minutesDiff(dateTimeValue2, dateTimeValue1) {
    var differenceValue = (dateTimeValue2.getTime() - dateTimeValue1.getTime()) / 1000;
    differenceValue /= 60;
    return Math.abs(Math.round(differenceValue));
 }

const denoiseSleepDataForVisualization = (sleepLogs) => {
  console.log('sleepLogs: ', sleepLogs);
  const denoisedSleepLogs = sleepLogs.map((log, i) => {
    if(sleepLogs[i+1] === undefined) return log;

    const thisEventTime = sleepLogs[i].time;
    const nextEventTime = sleepLogs[i+1].time;
    const minutes_diff_next = minutesDiff(nextEventTime, thisEventTime);
    const minutes_diff_prior = minutesDiff(thisEventTime, nextEventTime);
    // console.log('minutes_diff_next: ', minutes_diff_next);
    // console.log('minutes_diff_prior: ', minutes_diff_prior);

    if(minutes_diff_next > 800 || minutes_diff_prior > 800) { // bad data, probably not home, drop it
      // console.log('too long: ', minutes_diff_prior, ", ", minutes_diff_next);
      return null;
    } else if(minutes_diff_next <= 5 || minutes_diff_prior <=  5) {
      // console.log('too short: ', minutes_diff_prior, ", ", minutes_diff_next);
      return null;
    } else if(sleepLogs[i].awake == sleepLogs[i+1].awake) {
      // console.log('same');
      return null;
    }

    // if(log.time)
    return log;
  });
  const filteredDenoisedSleepLogs = denoisedSleepLogs.filter(log => log != null);
  return filteredDenoisedSleepLogs;
}
  

export default function Charts({ sleepLogs, forecast, setForecast }) {
    // if(!sleepLogs) return <React.Fragment></React.Fragment>;

    useMemo(() => {
      if(sleepLogs) {
          console.log("update")
          const beginningOfChartRange = new Date().addDays(-1);
          beginningOfChartRange.setHours(19,0,0,0);
          const endOfChartRange = new Date().addDays(1);
          endOfChartRange.setHours(6,0,0,0);
      
          // const chartEvents = eventsWithinRange(sleepLogs, beginningOfChartRange, endOfChartRange);
          const relevantChartEvents = sleepLogs.filter(log => {
              const logTime = log.time;
              return logTime > beginningOfChartRange && logTime < endOfChartRange;
          });
      
          // create and insert chart in body
          createChart(relevantChartEvents, beginningOfChartRange, endOfChartRange);
          const denoisedSleepLogs = denoiseSleepDataForVisualization(sleepLogs);
          createCrazyCircles(denoisedSleepLogs, forecast);    
      }
    }, [sleepLogs]);


    return (
        <div>
            <CardPane>
                <div id="chartContainer" style={{ width: '100%', height: '100%', overflow: 'auto', marginTop: '20px'}}>
                        <div style={{ width: '100%', height: '100%'}}>
                            <div id="graph"></div>
                        </div>
                </div>
            </CardPane>
            <CardPane>
                <canvas style={{margin: "10px", width: '95%'}} id="myCanvas"></canvas>
            </CardPane>
            {/* <div style={{textAlign: 'center', marginBottom: '150px'}} onClick={() => setForecast(!forecast)}>
                <div style={{marginBottom: '100px'}}>
                    <div style={{color: 'blue', borderRadius: '15px', textAlign: 'center', margin: 'auto', width: '70px', boxShadow: '0 0 30px 3px rgba(255,165,0,0.7)', backgroundColor: 'rgba(255,165,0,0.75)' }}>
                        <FontAwesomeIcon size="3x" icon={forecast ? faTree : faSeedling} />
                    </div>
                    <span style={{color: 'steelblue', textAlign: 'center', margin: 'auto', marginBottom: '100px'}}>{forecast ? 'Historical' : 'Forecast'}</span>
                </div>
            </div> */}
        </div>
    );
}