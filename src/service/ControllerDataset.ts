import * as tf from '@tensorflow/tfjs';
import {Tensor2D, Tensor3D, Tensor4D} from '@tensorflow/tfjs';

export default class ControllerDataset {

    public xs: Tensor4D | null = null;
    public ys: Tensor2D | null = null;

    constructor(public numClasses: number) {
        this.numClasses = numClasses;
    }

    /**
     * Adds an example to the controller dataset.
     * @param {Tensor} example representing the example of video frame
     * @param {number} label The label of the example. Should be a number.
     */
    addExample(example: Tensor3D, label: number) {
        const x = tf.tidy(() => example.expandDims<Tensor4D>(0));
        const y = tf.tidy(() => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses) as Tensor2D);

        if (this.xs === null || this.ys === null) {
            // This makes sure that if addExample() is called in a tf.tidy(), these Tensors will not get disposed.
            this.xs = tf.keep(x);
            this.ys = tf.keep(y);
        } else {
            const oldXs = this.xs;
            this.xs = tf.keep(oldXs.concat(x));

            const oldYs = this.ys;
            this.ys = tf.keep(oldYs.concat(y));

            oldXs.dispose();
            oldYs.dispose();
            x.dispose();
            y.dispose();
        }
    }

    reset() {
        if (this.xs && this.ys) {
            this.xs.dispose();
            this.xs = null;

            this.ys.dispose();
            this.ys = null;
        }
    }
}