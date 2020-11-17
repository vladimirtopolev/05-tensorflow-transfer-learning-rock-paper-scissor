import {useRef, useEffect, useState} from 'react';
import * as tfd from '@tensorflow/tfjs-data';
import {WebcamIterator} from '@tensorflow/tfjs-data/dist/iterators/webcam_iterator';


export default () => {
    const [webcamIterator, setWebcamIterator] = useState<WebcamIterator | null>(null);
    const [error, setError] = useState<String | null>(null);
    const videoRef = useRef<HTMLVideoElement>(null);

    useEffect(() => {
        (async () => {
            if (videoRef?.current) {
                try {
                    const webcam = await tfd.webcam(videoRef.current);
                    setWebcamIterator(webcam);
                } catch (e) {
                    setError(e.message);
                }
            }
        })();
    }, []);

    return {webcamIterator, error, videoRef}

}