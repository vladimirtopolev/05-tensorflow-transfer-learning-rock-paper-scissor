import React from 'react';
import ControllerDataset from '../service/ControllerDataset';
import styles from './WorkingArea.module.scss';
import WorkingAreaItem from './WorkinAreaItem';
import {WebcamIterator} from '@tensorflow/tfjs-data/dist/iterators/webcam_iterator';

type WorkingAreaProps = {
    numClasses: number,
    webcam: WebcamIterator | null,
    controllerDataset: ControllerDataset,
    activeLabel: number | undefined,
};

export default (props: WorkingAreaProps) => {
    return (
        <div className={styles.Items}>
            {Array.from({length: props.numClasses})
                .map((_, label) =>
                    <WorkingAreaItem label={label}
                                    key={label}
                                    isActive={props.activeLabel === label}
                                    {...props}
                    />)}
        </div>
    );
};