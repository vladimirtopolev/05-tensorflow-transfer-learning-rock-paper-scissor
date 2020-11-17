import {useEffect, useState} from 'react';
import {LayersModel, Shape, SymbolicTensor} from '@tensorflow/tfjs';
import * as tf from '@tensorflow/tfjs';

function buildNewHead(inputShape: Shape, numClasses: number): LayersModel {
    // Creates a 2-layer fully connected model
    return tf.sequential({
        layers: [
            tf.layers.flatten({
                name: 'flatten',
                inputShape: inputShape.slice(1),
            }),
            tf.layers.dense({
                name: 'hidden_dense_1',
                units: 100,
                activation: 'relu',
                kernelInitializer: 'varianceScaling',
                useBias: true
            }),
            // Layer 2. The number of units of the last layer should correspond
            // to the number of classes we want to predict.
            tf.layers.dense({
                name: 'softmax_classification',
                units: numClasses,
                kernelInitializer: 'varianceScaling',
                useBias: false,
                activation: 'softmax'
            })
        ]
    });
}

export default (pretrainedModel: LayersModel | undefined, numClasses: number) => {
    const [model, setModel] = useState<LayersModel>();

    useEffect(() => {
        if (pretrainedModel) {
            // find last convolutional layer by its name
            // it is a layer followed by layers which are responsible
            // for classification
            const truncatedLayer = pretrainedModel.getLayer('conv_pw_13_relu');
            const truncatedLayerOutput = truncatedLayer.output as SymbolicTensor;

            // freeze all layers of MobileNet
            for (const layer of pretrainedModel.layers) {
                layer.trainable = false;
            }

            const transferHead = buildNewHead(truncatedLayerOutput.shape, numClasses);
            const newOutput = transferHead.apply(truncatedLayerOutput) as SymbolicTensor;

            const model = tf.model({
                inputs: pretrainedModel.inputs,
                outputs: newOutput
            });

            model.compile({
                optimizer: tf.train.adam(0.0001),
                loss: 'categoricalCrossentropy'
            });
            setModel(model);
        }
    }, [pretrainedModel]);

    return model;
}