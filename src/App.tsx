import React, {useEffect, useRef, useState} from 'react';
import useUploadModel from './hook/useUploadModel';
import useWebcamIterator from './hook/useWebcamIterator';
import WorkingArea from './components/WorkingArea';
import useNewModelTopology from './hook/useNewModelTopology';
import ControllerDataset from './service/ControllerDataset';
import * as tf from '@tensorflow/tfjs';
import {Tensor4D, Tensor} from '@tensorflow/tfjs';
import {Container, Button, Grid, Box} from '@material-ui/core';

const NUM_CLASSES = 3;

enum TRAINING_STATES {
    NOT_TRAINED,
    TRAINING,
    TRAINED
}

function App() {
    const uploadedModel = useUploadModel();
    const model = useNewModelTopology(uploadedModel, NUM_CLASSES);
    const {videoRef, webcamIterator} = useWebcamIterator();

    const [loss, setLoss] = useState<string | null>(null);
    const controllerDataset = useRef(new ControllerDataset(NUM_CLASSES));

    const [trainingState, changeTrainingState] = useState<TRAINING_STATES>(TRAINING_STATES.NOT_TRAINED);
    const [activeLabel, setActiveLabel] = useState<number>();

    const train = async () => {
        if (model
            && controllerDataset.current.xs !== null
            && controllerDataset.current.ys !== null) {
            const {current: {xs, ys}} = controllerDataset;
            model.fit(xs, ys, {
                batchSize: 24,
                epochs: 20,
                shuffle: true,
                callbacks: {
                    onBatchEnd: async (batch, logs) => {
                        setLoss(() => (logs?.loss as number).toFixed(5));
                    },
                    onTrainEnd: () => {
                        changeTrainingState(() => TRAINING_STATES.TRAINED);
                    }
                }
            });
        }
    };

    const predict = async () => {
        if (webcamIterator && model) {
            const image = await webcamIterator.capture();
            const processedImage = tf.tidy<Tensor4D>(
                () => image.expandDims().toFloat().div(127).sub(1)
            );
            const label = tf.argMax(model.predict(processedImage) as Tensor, 1).dataSync();
            setActiveLabel(() => label[0]);
            image.dispose();
            processedImage.dispose();
            requestAnimationFrame(() => predict());
        }
    };

    useEffect(() => {
        if (trainingState === TRAINING_STATES.TRAINED) {
            requestAnimationFrame(() => predict());
        }
    }, [trainingState]);

    return (
        <Box m={2}>
            <Container maxWidth={'lg'}>
                <Grid container justify={'center'}>
                    <Grid item>
                        <video ref={videoRef} width={224} height={224}/>
                        <Box>
                            <Button
                                color="primary"
                                variant="contained"
                                disabled={trainingState === TRAINING_STATES.TRAINING}
                                onClick={() => {
                                    if (trainingState !== TRAINING_STATES.TRAINING) {
                                        changeTrainingState(() => TRAINING_STATES.TRAINING);
                                        train();
                                    }
                                }}>
                                Train
                            </Button>
                            {loss && <Box component={'span'} m={1}>Loss: {loss}</Box>}
                        </Box>
                    </Grid>
                </Grid>
                <Grid container justify={'center'}>
                    <Box m={2}>
                        <WorkingArea numClasses={NUM_CLASSES}
                                     webcam={webcamIterator}
                                     activeLabel={activeLabel}
                                     controllerDataset={controllerDataset.current}
                        />
                    </Box>
                </Grid>
            </Container>
        </Box>
    );
}

export default App;
