// @ts-nocheck
import React, { useEffect, useState } from "react";
import ReactSpeedometer from "react-d3-speedometer"
import CircularProgress from '@mui/material/CircularProgress';
import { VideoFeedTypeEnum } from './VideoFeed';

const REFETCH_CLASSIFICATION_MS = 2000;

type RetrainProps = {
  videoFeedType: VideoFeedTypeEnum
}
export default function Retrain({ videoFeedType }: RetrainProps) {

    const [initiallyLoading, setInitiallyLoading] = useState(true);
    const [awakeProba, setAwakeProba] = useState(0.5);
    const [voteReasons, setVoteReasons] = useState([]);

    useEffect(() => {
        const intervalId = setInterval(async () => { 
            const [awakeProba, ...reasons] = (await (await fetch(`http://${process.env.REACT_APP_BACKEND_IP}/getResultAndReasons`)).text()).split(',');
            console.log('babyProba: ', awakeProba);
            console.log('reasons: ', reasons);
            setVoteReasons(reasons as any);
            setInitiallyLoading(false);
            setAwakeProba(parseFloat(awakeProba));
        }, REFETCH_CLASSIFICATION_MS)

        return () => clearInterval(intervalId);
    }, [])
  
  if(voteReasons.length > 0 && false) { // never

    const texts = voteReasons;

    const elts = {
      text1: document.getElementById("text1"),
      text2: document.getElementById("text2")
    };

    console.log('text: ', elts.text1.textContent);      
    
    // Controls the speed of morphing.
    const morphTime = 2;
    const cooldownTime = 0.4;

    let textIndex = texts.length - 1;
    let time = new Date();
    let morph = 0;
    let cooldown = cooldownTime;

    elts.text1.textContent = texts[textIndex % texts.length];
    elts.text2.textContent = texts[(textIndex + 1) % texts.length];
    console.log('elts.text1.textContent: ', elts.text1.textContent);
    console.log('elts.text2.textContent: ', elts.text2.textContent);

    // Animation loop, which is called every frame.
    function animate() {
      requestAnimationFrame(animate);
      
      let newTime = new Date();
      let shouldIncrementIndex = cooldown > 0;
      let dt = (newTime - time) / 1000;
      time = newTime;
      
      cooldown -= dt;
      
      if (cooldown <= 0) {
        if (shouldIncrementIndex) {
          textIndex++;
        }
        
        doMorph();
      } else {
        doCooldown();
      }
    }

    function doMorph() {
      morph -= cooldown;
      cooldown = 0;
      
      let fraction = morph / morphTime;
      
      if (fraction > 1) {
        cooldown = cooldownTime;
        fraction = 1;
      }
      
      setMorph(fraction);
    }

    // A lot of the magic happens here, this is what applies the blur filter to the text.
    function setMorph(fraction) {
      // fraction = Math.cos(fraction * Math.PI) / -2 + .5;
      
      elts.text2.style.filter = `blur(${Math.min(8 / fraction - 8, 100)}px)`;
      elts.text2.style.opacity = `${Math.pow(fraction, 0.4) * 100}%`;
      
      fraction = 1 - fraction;
      elts.text1.style.filter = `blur(${Math.min(8 / fraction - 8, 100)}px)`;
      elts.text1.style.opacity = `${Math.pow(fraction, 0.4) * 100}%`;
      
      elts.text1.textContent = texts[textIndex % texts.length];
      elts.text2.textContent = texts[(textIndex + 1) % texts.length];
    }

    function doCooldown() {
      morph = 0;
      elts.text2.style.filter = "";
      elts.text2.style.opacity = "100%";
      
      elts.text1.style.filter = "";
      elts.text1.style.opacity = "0%";
    }

    animate();
  }


  const wakeReasons = ['Eyes Open', 'Moving', 'Movement', 'No baby present'];
  // const sleepReasons = ['Eyes Closed', 'Not moving', 'Baby present', 'Baby not moving'];
  const styledVoteReasons = voteReasons.map((reason, i) => (
    <span style={{ color: wakeReasons.includes(reason) ? 'orange' : '#007FFF'}}>{reason}{i < voteReasons.length-1 ? ', ' : ''}</span>
  ));

  const activityText = videoFeedType === VideoFeedTypeEnum.ML && styledVoteReasons.length > 0 ? (
    <div style={{ width: '90%', display: 'flex', justifyContent: 'center', marginLeft: '5%' }}>
      <span style={{ fontSize: '18px', float: 'left' }}>
        <b>Activity: </b>
      </span>
      <span style={{ fontSize: '18px' }}>
        {styledVoteReasons}
      </span>
    </div>
  ) : <></>;

  const percentageAwakeProba = awakeProba * 100;
  const speedometerValue = percentageAwakeProba >= 98 ? 98 : percentageAwakeProba <= 2 ? 2 : percentageAwakeProba;
  return (
      <div style={{ width: '100%' }}>
        <h3 style={{ textAlign: 'center', marginBottom: '-5px' }}>
          Your Baby's Sleepometer
        </h3>
        {initiallyLoading && <div style={{ marginTop: '25px' }}><CircularProgress style={{color: 'orange'}} /></div> }
        <div style={{ width: '75%', marginLeft: '12.5%' }}>
          <ReactSpeedometer
            height={200}
            width={200}
            fluidWidth={true}
            segmentColors={['blue', 'orange']}
            minValue={0}
            maxValue={100}
            segments={2}
            needleHeightRatio={0.7}
            value={speedometerValue}
            valueTextFontSize={0}
            currentValueText={'This is what the app thinks'}
            customSegmentLabels={[
              {
                text: 'Asleep',
                position: 'INSIDE',
                color: 'orange',
              },
              {
                text: 'Awake',
                position: 'INSIDE',
                color: 'blue',
              },
            ]}
            ringWidth={35}
            needleTransitionDuration={3333}
            needleTransition="easeElastic"
            needleColor={'steelblue'}
            textColor={'steelblue'}
          />
        </div>

        {activityText}

      </div>
  );
}