import React, {useState, useRef} from 'react';
import * as tf from '@tensorflow/tfjs';
import {Tensor3D} from '@tensorflow/tfjs';
import styles from './WorkingArea.module.scss';
import {WebcamIterator} from '@tensorflow/tfjs-data/dist/iterators/webcam_iterator';
import ControllerDataset from '../service/ControllerDataset';

type WorkingAreaItemProps = {
    label: number,
    webcam: WebcamIterator | null,
    controllerDataset: ControllerDataset,
    isActive: boolean
};

export default ({label, controllerDataset, webcam, isActive}: WorkingAreaItemProps) => {
    const mode = useRef<boolean>(false);
    const [count, setCount] = useState(0);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const processFrame = async () => {
        while (mode.current) {
            if (webcam && controllerDataset && canvasRef.current) {
                // Captures a frame from the webcam and normalizes it between -1 and 1.
                const image = await webcam.capture();
                const processedImage = tf.tidy<Tensor3D>(
                    () => image.toFloat().div(127).sub(1)
                );
                controllerDataset.addExample(processedImage, label);
                await tf.browser.toPixels(image, canvasRef.current);
                setCount(count => count + 1);
                image.dispose();
                processedImage.dispose();
            }
        }
    };

    return (
        <div>
            <div key={label} className={[
                styles.Item,
                mode.current || isActive  ? styles.Item_isActive : ''
            ].join(' ')}>
                <div className={styles.Item__container}>
                    <canvas ref={canvasRef}
                            className={styles.Item__canvas}
                            width={224}
                            height={224}/>
                    <button
                        className={styles.Item__btn}
                        onMouseDown={() => {
                            mode.current = true;
                            processFrame();
                        }}
                        onMouseUp={() => {
                            mode.current = false;
                        }}
                    >
                        Add example
                    </button>
                </div>
            </div>
            <div>{count} examples</div>
        </div>
    );
}