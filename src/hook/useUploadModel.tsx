import {useEffect, useState} from 'react';
import * as tf from '@tensorflow/tfjs';
import {LayersModel} from '@tensorflow/tfjs';

const MODEL_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

export default () => {
    const [model, setModel] = useState<LayersModel>();

    useEffect(() => {
        (async function init() {
            const model = await tf.loadLayersModel(MODEL_URL);
            setModel(() => model);
            model.summary();
        })();
    }, []);

    return model;
}