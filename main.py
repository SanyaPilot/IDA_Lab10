from PIL import Image
from PIL.Image import Dither
import numpy as np
import os
from math import exp
from collections import defaultdict
from typing import Callable

EPOCH_COUNT = 250
TWO_LAYER_EPOCH_COUNT = 30


class Layer:
    def __init__(self, func: Callable[[float], float], inp_size: int, layer_size: int):
        self._func = func
        self._size = layer_size
        # Weights matrix (for layer_size neurons, with inp_size weight vector size)
        self._weights = np.random.rand(layer_size, inp_size) - 0.5
        # Biases vector (for layer_size neurons)
        self._biases = np.zeros(layer_size)

    def process(self, inp: np.ndarray) -> np.ndarray:
        return np.array([self._func(np.dot(inp, self._weights[i]) + self._biases[i]) for i in range(self._size)])

    def train(self, inputs: np.ndarray, labels: np.ndarray, mu: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        results = np.array([self.process(inp) for inp in inputs])
        deltas = []
        for k in range(self._size):
            cur_deltas = np.array([
                (res[k] - label[k]) * res[k] * (1 - res[k])
                for res, label in zip(results, labels)
            ])
            for j in range(self._weights.shape[1]):
                self._weights[k, j] -= mu * sum([delta * inp[j] for delta, inp in zip(cur_deltas, inputs)])

            self._biases[k] -= mu * sum(cur_deltas)
            deltas.append(cur_deltas)

        return self._weights, self._biases, np.array(deltas)

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def biases(self) -> np.ndarray:
        return self._biases


class NeuralNetwork:
    def __init__(self, *layers: Layer):
        self._layers: tuple[Layer, ...] = layers

    def process(self, inp: np.ndarray) -> list[np.ndarray]:
        results = []
        for layer in self._layers:
            if not results:
                results.append(layer.process(inp))
            else:
                results.append(layer.process(results[-1]))

        return results

    def train(self, inputs: np.ndarray, labels: np.ndarray, mu: float = 0.05):
        input_results = [self.process(inp) for inp in inputs]
        layer_results: list[list | None] = [None] * len(self._layers)
        for result in input_results:
            for l_id in range(len(result)):
                if not layer_results[l_id]:
                    layer_results[l_id] = []

                layer_results[l_id].append(result[l_id])

        hidden_labels = []
        for a in range(len(inputs)):
            hidden_labels.append([None] * (len(self._layers) - 1))
            for i in range(len(self._layers) - 2, -1, -1):
                # Train i + 1 layer
                weights, _, deltas = self._layers[i + 1].train(layer_results[i], labels if i + 1 == len(self._layers) - 1 else hidden_labels[a][i], mu=mu)
                hidden_labels[a][i] = [layer_results[i][a][m] - sum([deltas[k] * weights[k, m] for k in range(len(weights))]) for m in range(len(layer_results[i][a]))]
            # работает или нет хз...


def load_dataset(path: str) -> dict[int, list[np.ndarray]]:
    files = os.listdir(path)
    arrays = defaultdict(list)
    for file in files:
        with Image.open(os.path.join(path, file)) as im:
            bw_im = im.convert('1', dither=Dither.NONE)
            arrays[int(file.split('_')[0])].append(np.array(bw_im, dtype='uint8').flatten())

    return arrays


def sigmoid(x) -> float:
    return 1 / (1 + exp(-x))


def main():
    dataset_5x5 = load_dataset('dataset/5x5')
    dataset_10x10 = load_dataset('dataset/10x10')

    layer_5x5 = Layer(sigmoid, 25, 10)
    inputs, labels = [], []
    for label, images in dataset_5x5.items():
        for im in images:
            l_arr = np.zeros(10)
            l_arr[label] = 1
            labels.append(l_arr)
            inputs.append(im)

    for i in range(EPOCH_COUNT):
        layer_5x5.train(np.array(inputs), np.array(labels))

    print(f"Trained with {EPOCH_COUNT} epochs!\nW: {layer_5x5.weights}\n\nb: {layer_5x5.biases}")

    hit_count = 0
    all_count = 0
    for label, images in dataset_5x5.items():
        for im in images:
            all_count += 1
            probs = layer_5x5.process(im)
            res = None
            max_prob = 0
            for i in range(len(probs)):
                if probs[i] > max_prob:
                    max_prob = probs[i]
                    res = i

            print(f"Label: {label}, result: {res}")
            if res == label:
                hit_count += 1

    print(f"Accuracy: {hit_count / all_count * 100}%")

    nn_10x10 = NeuralNetwork(Layer(sigmoid, 100, 32), Layer(sigmoid, 32, 10))
    inputs, labels = [], []
    for label, images in dataset_10x10.items():
        l_arr = np.zeros(10)
        l_arr[label] = 1
        labels.append(l_arr)
        inputs.append(images[0])
        # Пьяные стратки, не судите строго
        # докидываем еще кусок из картинок с именем X_2.png
        if label % 2 == 0:
            inputs.append(images[1])
            labels.append(l_arr)

    for i in range(TWO_LAYER_EPOCH_COUNT):
        nn_10x10.train(np.array(inputs), np.array(labels))

    print(f"Trained with {TWO_LAYER_EPOCH_COUNT} epochs!")

    hit_count = 0
    all_count = 0
    for label, images in dataset_10x10.items():
        for im in images:
            all_count += 1
            probs = nn_10x10.process(im)[-1]
            res = None
            max_prob = 0
            for i in range(len(probs)):
                if probs[i] > max_prob:
                    max_prob = probs[i]
                    res = i

            print(f"Label: {label}, result: {res}")
            if res == label:
                hit_count += 1

    print(f"Accuracy: {hit_count / all_count * 100}%")


if __name__ == '__main__':
    main()
