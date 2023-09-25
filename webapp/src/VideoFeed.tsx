import React, { Fragment, useState, useEffect } from "react";
import { ToggleButtonGroup, ToggleButton, styled, Alert, AlertTitle } from '@mui/material'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faMicrochip, faEye, faBaby, faXmark, faCropSimple } from '@fortawesome/free-solid-svg-icons'
import LinearProgress, { linearProgressClasses } from '@mui/material/LinearProgress';
import { ProgressBar } from  'react-loader-spinner'
import { CroppableVideoFeed } from './CroppableVideoFeed';

const BorderLinearProgress = styled(LinearProgress)<{ awake: boolean }>(({ theme, awake }) => ({
  marginTop: '8px',
  marginBottom: '5px',
  width: '100%',
  height: 12,
  // margin: 'auto',
  borderRadius: 5,
  [`&.${linearProgressClasses.colorPrimary}`]: {
    // backgroundColor: 'orange'
    background: awake ? 'linear-gradient(-45deg, #ffc400, #ffa600, #ff9900, #ff6f00)' : 'rgba(255, 165, 0, .45)',
    backgroundSize: awake ? '400% 400%' : 'auto',
    // boxShadow: awake ? '2px 0px 10px 2px #ff9900' : 'none',
    animation: awake ? 'gradient 2s ease infinite' : 'none',
  },
  [`& .${linearProgressClasses.bar}`]: {
    borderRadius: 5,
    // box-shadow: 120px 80px 40px 20px #0ff;
    // backgroundColor: 'blue',
    background: awake ? 'rgba(0, 0, 255, .9)' : 'linear-gradient(-45deg, #004cff, #0032ff, #0000ff)',
    backgroundSize: awake ? 'auto' : '400% 400%',
    // boxShadow: awake ? 'none' : '2px 0px 10px 2px #0000ff',
    animation: awake ? 'none' : 'gradient 2s ease infinite',
  },
}));

const REFETCH_CLASSIFICATION_MS = 3000;

export enum VideoFeedTypeEnum {
  RAW = 'raw',
  ML = 'ml',
  CROP = 'crop',
}

type VideoFeedProps = {
  modelProba: any;
  setModelProba: any;
  videoFeedType: VideoFeedTypeEnum;
  setVideoFeedType: any;
}

export default function VideoFeed({ modelProba, setModelProba, videoFeedType, setVideoFeedType }: VideoFeedProps) {

  const [bodyFound, setBodyFound] = useState(false)
  const videoFeed = videoFeedType === VideoFeedTypeEnum.ML ? `http://${process.env.REACT_APP_BACKEND_IP}/video_feed/processed` : `http://${process.env.REACT_APP_BACKEND_IP}/video_feed/raw`;

  useEffect(() => {
    const intervalId = setInterval(async () => { 
        const [presentProba, _notPresentProba, _time, bodyFound] = (await (await fetch(`http://${process.env.REACT_APP_BACKEND_IP}/getClassificationProbabilities`)).text()).split(',');
        console.log('modelProba: ', presentProba);
        console.log("bodyFound: ", bodyFound)
        setBodyFound(bodyFound.toLowerCase() === "true");
        if(presentProba !== '') {
          setModelProba(parseFloat(presentProba));
        }
    }, REFETCH_CLASSIFICATION_MS)

    return () => clearInterval(intervalId);
  }, [])

  const [retraining, setRetraining] = useState(false);
  const [success, setSuccess] = useState(false);

  const successNotification = (
    <Alert severity="success">
      <AlertTitle>Sleepometer grows stronger.</AlertTitle>
    </Alert>
  );

  const sendRetrainingRequest = async (classification: any) => {
    if(retraining) return;
    setRetraining(true);
    console.log('Retraining...');
    const trainingResult = await fetch(`http://${process.env.REACT_APP_BACKEND_IP}/retrainWithNewSample/${classification}`);
    console.log("res: ", trainingResult);
    setRetraining(false);
    setSuccess(true);
    setTimeout(() => {
      setSuccess(false);
    }, 5000)
  }

  const CustomToggle = styled(ToggleButton)({
    color: '#4682B4'
  })
  console.log('modelProba: ', modelProba)
  const requireUserToSetBounds = modelProba instanceof String && modelProba.includes('Bounds not set') || Number.isNaN(modelProba);
  console.log('requireUserToSetBounds: ', requireUserToSetBounds);

  const SharedImgElement: JSX.Element = (<img style={{ borderRadius: '25px' }} src={videoFeed} width="90%" height="90%" />);

  const videoElement = videoFeedType === VideoFeedTypeEnum.CROP ?
    <CroppableVideoFeed videoFeed={videoFeed} SharedImgElement={SharedImgElement} retraining={retraining} setRetraining={setRetraining} success={success} setSuccess={setSuccess} />
    : SharedImgElement

  const modelHeaderText = bodyFound ? "Anatomical features detected, ignoring custom presence detection AI" : "Using custom presence detection AI";
  return (

    <div style={{ width: '100%', marginTop: '10px' }}>
        <div>
          <div style={{ width: '100%', position: 'relative' }}>
            <h3 style={{ position: 'absolute', left: '10%', zIndex: '1' }}><span style={{ color: 'red' }}>🔴 Live</span></h3>

            {videoElement}

          </div>
          {videoFeedType === VideoFeedTypeEnum.ML ? (
          <div>
            <div style={{ width: '100%', display: 'flex', alignItems: 'center', flexDirection: 'column', justifyContent: 'center', paddingBottom: '15px' }}>
              <span style={{ marginTop: '8px', width: '90%' }}><b>{modelHeaderText}</b></span>
              <span style={{ marginBottom: '24px', width: '90%' }}><i>If incorrect, use buttons to retrain</i></span>
              {success && (
                <div style={{ marginBottom: '25px'}}>
                  {successNotification}
                </div>
              )}
              <div style={{ display: 'flex', flexDirection: 'row', marginBottom: '15px' }}>
                  {retraining &&
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <ProgressBar
                        height="80"
                        width="80"
                        ariaLabel="progress-bar-loading"
                        wrapperClass="progress-bar-wrapper"
                        borderColor = 'orange'
                        barColor = 'steelblue'
                      />
                      <p style={{ margin: 0, marginTop: '-20px' }}>Training...</p>
                    </div>
                  }
              </div>
              
              <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between', width: '90%', opacity: bodyFound ? '0.5' : '1.0', }}>
                <div style={{ display: 'flex', flexDirection: 'column', opacity: modelProba > 0.5 ? .3 : 1 }}>
                  {/* <FontAwesomeIcon size="3x" icon={faBaby} color="blue"/>
                  <p style={{ margin: '0' }}>Baby present</p> */}
                  <div onClick={() => (modelProba <= 0.5 && !bodyFound) ? sendRetrainingRequest('baby') : null} style={{color: 'orange', borderRadius: '15px', textAlign: 'center', margin: 'auto', width: '60px', boxShadow: '1px 1px 10px 1px rgba(0,0,0,0.7)', backgroundColor: 'rgba(0,0,255,0.75)' }}>
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                      <FontAwesomeIcon size="2x" icon={faBaby} />
                      <p style={{ margin: '0', marginBottom: '8px' }} >Yes</p>
                    </div>
                  </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', width: '55%', justifyContent: 'center' }}>
                  <span >Is your baby present? <b className='linear-wipe' style={{
                      'WebkitBackgroundClip': 'text',
                      background: modelProba <= 0.5 ? 'linear-gradient(-45deg, #ffc400 20%, #ffa600 40%, #ff9900 60%, #ff6f00 80%)' : 'linear-gradient(-45deg, rgb(150, 150, 255), rgb(100, 150, 255), rgb(50, 50, 255))'
                    }}>{modelProba <= 0.5 ? 'No' : 'Yes'}</b></span>
                  <BorderLinearProgress variant="determinate" value={modelProba*100} awake={modelProba <= 0.5} />
                  <span>AI Confidence</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', opacity: modelProba < 0.5 ? .3 : 1 }}>
                  {/* <FontAwesomeIcon size="3x" icon={faXmark} color="orange"/>
                  <p style={{ margin: '0' }}>Baby not present</p> */}
                  <div onClick={() => (modelProba >= 0.5 && !bodyFound) ? sendRetrainingRequest('no_baby') : null} style={{color: 'blue', borderRadius: '15px', textAlign: 'center', margin: 'auto', width: '60px', boxShadow: '1px 1px 10px 1px rgba(0,0,0,0.7)', backgroundColor: 'rgba(255,165,0,0.75)' }}>
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                      <FontAwesomeIcon size="2x" icon={faXmark} />
                      <p style={{ margin: '0', marginBottom: '8px' }}>No</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
        </div>
          
          )
        : <Fragment></Fragment>}
      </div>


      
      <div style={{ width: '100%', marginTop: '20px', marginBottom: '10px', display: 'flex', flexDirection: 'row', justifyContent: 'center' }}>
        {/* <h3><span style={{ color: 'red' }}>🔴 Live</span></h3> */}

        <ToggleButtonGroup
          value={videoFeedType}
          exclusive
          onChange={(_e, videoFeedType) => setVideoFeedType(videoFeedType)}
          color="primary"
        >
          <CustomToggle disabled={requireUserToSetBounds} value={VideoFeedTypeEnum.RAW}>
             <FontAwesomeIcon size="3x" icon={faEye} />
          </CustomToggle>
          <CustomToggle disabled={requireUserToSetBounds} value={VideoFeedTypeEnum.ML}>
            <FontAwesomeIcon size="3x" icon={faMicrochip} />
          </CustomToggle>
          <CustomToggle value={VideoFeedTypeEnum.CROP}>
            <FontAwesomeIcon size="3x" icon={faCropSimple} />
          </CustomToggle>
        </ToggleButtonGroup>

      </div>
    </div>
  );
}
