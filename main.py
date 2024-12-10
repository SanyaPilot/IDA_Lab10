from PIL import Image
from PIL.Image import Dither
import numpy as np
import os
from math import exp
from collections import defaultdict
from typing import Callable
from datetime import datetime

EPOCH_COUNT = 250
TWO_LAYER_EPOCH_COUNT = 100


class Layer:
    def __init__(self, func: Callable[[float], float], inp_size: int, layer_size: int):
        """
        :param func: Activation function
        :param inp_size: Number of inputs (number of neurons in the previous layer, or input data size)
        :param layer_size: Number of neurons in this layer
        """
        self._func = func
        self._size = layer_size
        # Weights matrix (for layer_size neurons, with inp_size weight vector size)
        self._weights = np.random.rand(layer_size, inp_size) - 0.5
        # Biases vector (for layer_size neurons)
        self._biases = np.zeros(layer_size)

    def process(self, inp: np.ndarray) -> np.ndarray:
        """
        Calculate layer out vector
        :param inp: NumPy vector of input values
        :return: NumPy vector of results
        """
        return np.array([self._func(np.dot(inp, self._weights[i]) + self._biases[i]) for i in range(self._size)])

    def train(self, inputs: np.ndarray, labels: np.ndarray, mu: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform one training cycle using delta rule
        :param inputs: NumPy matrix of inputs (each row - one input vector)
        :param labels: NumPy vector of labels for supplied inputs
        :param mu: Gradient descent coefficient
        :return: Tuple containing a weights matrix, a biases vector and a list of deltas
        """
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
        """
        Calculate layer results, moving forward
        :param inp: NumPy vector of input values
        :return: List of layer results (final result is the last)
        """
        results = []
        for layer in self._layers:
            if not results:
                results.append(layer.process(inp))
            else:
                results.append(layer.process(results[-1]))

        return results

    def train(self, inputs: np.ndarray, labels: np.ndarray, mu: float = 0.05):
        """
        Perform one training cycle using backpropagation
        :param inputs: NumPy matrix of inputs (each row - one input vector)
        :param labels: NumPy vector of labels for supplied inputs
        :param mu: Gradient descent coefficient
        """
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
                # Get hidden labels for i layer using a nice formula
                hidden_labels[a][i] = [layer_results[i][a][m] - sum([deltas[k] * weights[k, m] for k in range(len(weights))]) for m in range(len(layer_results[i][a]))]


def sigmoid(x) -> float:
    return 1 / (1 + exp(-x))


def load_dataset(path: str) -> dict[int, list[np.ndarray]]:
    files = os.listdir(path)
    arrays = defaultdict(list)
    for file in files:
        with Image.open(os.path.join(path, file)) as im:
            bw_im = im.convert('1', dither=Dither.NONE)
            arrays[int(file.split('_' if '_' in file else '.')[0])].append(np.array(bw_im, dtype='uint8').flatten())

    return arrays


def dataset_to_nn_input(dataset: dict[int, list[np.ndarray]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    inputs, labels = [], []
    for label, images in dataset.items():
        for im in images:
            l_arr = np.zeros(10)
            l_arr[label] = 1
            labels.append(l_arr)
            inputs.append(im)

    return inputs, labels


def test_network(test_dataset: dict[int, list[np.ndarray]], network: NeuralNetwork | Layer):
    hit_count = 0
    all_count = 0
    for label, images in test_dataset.items():
        for im in images:
            all_count += 1
            probs = network.process(im)
            if isinstance(network, NeuralNetwork):
                probs = probs[-1]

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


def main():
    dataset_5x5 = load_dataset('dataset/5x5')
    dataset_10x10 = load_dataset('dataset/10x10')

    print("===== One layer network =====")
    layer_5x5 = Layer(sigmoid, 25, 10)
    inputs, labels = dataset_to_nn_input(dataset_5x5)
    start_time = datetime.now()
    for _ in range(EPOCH_COUNT):
        layer_5x5.train(np.array(inputs), np.array(labels))

    elapsed = datetime.now() - start_time
    print(f"Trained with {EPOCH_COUNT} epochs!\nW: {layer_5x5.weights}\n\nb: {layer_5x5.biases}")
    print(f"Training took {elapsed}\n")

    test_network(load_dataset('dataset/testing/5x5'), layer_5x5)

    print("\n\n===== Two-layer network =====")
    nn_10x10 = NeuralNetwork(Layer(sigmoid, 100, 32), Layer(sigmoid, 32, 10))
    inputs, labels = dataset_to_nn_input(dataset_10x10)
    start_time = datetime.now()
    for _ in range(TWO_LAYER_EPOCH_COUNT):
        nn_10x10.train(np.array(inputs), np.array(labels))

    elapsed = datetime.now() - start_time
    print(f"Trained with {TWO_LAYER_EPOCH_COUNT} epochs!")
    print(f"Training took {elapsed}\n")

    test_network(load_dataset('dataset/testing/10x10'), nn_10x10)


if __name__ == '__main__':
    main()
