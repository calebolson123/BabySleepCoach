// @ts-nocheck
import React, { useEffect, useRef, useState } from 'react';
import { Annotorious } from '@recogito/annotorious';
import { Button } from '@mui/material'
import { ProgressBar } from  'react-loader-spinner'

import '@recogito/annotorious/dist/annotorious.min.css';

type CroppableVideoFeedProps = {
    videoFeed: string;
    SharedImgElement: JSX.Element;
    retraining: boolean;
    setRetraining: any;
    success: boolean;
    setSuccess: any;
}

export function CroppableVideoFeed({ videoFeed, SharedImgElement, retraining, setRetraining, success, setSuccess }: CroppableVideoFeedProps) {
  const imgEl: any = useRef();
  const [anno, setAnno] = useState<any>();
  const [boundingBox, setBoundingBox] = useState<string | undefined>();
  console.log('boundingBox: ', boundingBox);

  useEffect(() => {
    let annotorious: any = null;

    if (imgEl.current) {
      annotorious = new Annotorious({
        image: imgEl.current,
        disableEditor: true,
        handleRadius: 14,
      });

      annotorious.on('createSelection', (annotation: any) => {
        console.log('createSelection', annotation);
        console.log(annotation.target.selector.value);
        const bounds = annotation.target.selector.value;
        setBoundingBox(bounds);
      });

      annotorious.on('changeSelectionTarget', (annotation: any) => {
        console.log('changeSelectionTarget', annotation);
        console.log(annotation.selector.value);
        setBoundingBox(annotation.selector.value);
      });

      annotorious.on('cancelSelected', (annotation: any) => {
        console.log("DELETEEEEEE")
        setBoundingBox(undefined);
      });
    }

    // Keep current Annotorious instance in state
    setAnno(annotorious);

    // Cleanup: destroy current instance
    return () => annotorious.destroy();
  }, []);

  const setAIFocusRegion = async () => {
    console.log('anno: ', anno);
    anno?.cancelSelected();
    anno?.clearAnnotations();
    // const newAnno = new Annotorious({
    //   image: imgEl.current,
    //   disableEditor: true,
    //   handleRadius: 14,
    // });
    // setAnno(newAnno);
    // console.log('set e: ', boundingBox);
    const yup = boundingBox!.split(':')[1]
    // console.log('yup: ', yup);
    setRetraining(true);
    const result = await fetch(`http://${process.env.REACT_APP_BACKEND_IP}/setAIFocusRegion/${yup}`);
    setRetraining(false);
  };

  const resetAIFocusRegion = async () => {
    anno?.cancelSelected();
    anno?.clearAnnotations();
    const result = await fetch(`http://${process.env.REACT_APP_BACKEND_IP}/setAIFocusRegion/reset`);
    // console.log('result: ', result);
  };

  const instructionText = "Drag on image to crop focus region for the AI";
  const tipText = "Tip: only capture the sleeping area";
  return (
    <div id="croppableDivId">
        <img style={{ borderRadius: '25px', display: 'initial !important' }} ref={imgEl} src={videoFeed} width="90%" height="90%" />
        <div style={{ marginTop: '10px', width: '100%', display: 'flex !important', alignItems: 'center', flexDirection: 'column', justifyContent: 'center' }}>
            <span><b>{instructionText}</b></span><br></br>
            <span><i>{tipText}</i></span>
              <div style={{ display: 'flex', flexDirection: 'row', marginTop: '-20px', marginBottom: '15px', justifyContent: 'center' }}>
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
            <div style={{ marginTop: '20px', marginBottom: '10px' }}>
                <Button onClick={setAIFocusRegion} disabled={boundingBox === undefined} variant="contained" color="success">Set</Button>
                <Button onClick={resetAIFocusRegion} style={{ marginLeft: '10%' }} variant="contained" color="error">Reset</Button>
            </div>
        </div>
    </div>
  );
}