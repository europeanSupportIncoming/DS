import random
from typing import List
from autodiff.neural_net import MultiLayerPerceptron
from autodiff.scalar import Scalar
import numpy as np


class MLPClassifierOwn:
    def __init__(
        self,
        num_epochs=5,
        alpha=0.0,
        batch_size=32,
        hidden_layer_sizes=(100,),
        random_state=0,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.num_classes = None
        self.model = None

    @staticmethod
    def softmax(z: List[Scalar]) -> List[Scalar]:
        """
        Returns the softmax of the given list of Scalars (as another list of Scalars).

        :param z: List of Scalar values
        """
        normalizer = np.sum(np.array([np.exp(zi) for zi in z]))
        return [np.exp(zi)/normalizer for zi in z]

    @staticmethod
    def sigmoid(z: Scalar) -> Scalar:
        """
        Returns the sigmoid of the given Scalar (as another Scalar).

        :param z: Scalar
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def multiclass_cross_entropy_loss(y_true: int, probs: List[Scalar]) -> Scalar:
        """
        Returns the multi-class cross-entropy loss for a single sample (as a Scalar).

        :param y_true: True class index (0-based)
        :param probs: List of Scalar values, representing the predicted probabilities for each class
        """
        return -np.log(probs[y_true])


    @staticmethod
    def binary_cross_entropy_loss(y_true: int, prob: Scalar) -> Scalar:
        """
        Returns the binary cross-entropy loss for a single sample.

        :param y_true: 0 or 1
        :param prob: Scalar between 0 and 1, representing the probability of the positive class
        """
        q = 1 - prob
        return (-np.log(prob) - np.log(q))

    def l2_regularization_term(self) -> Scalar:
        """
        Returns the L2 regularization term for the model. This is added to the loss if self.alpha > 0.

        Compute the sum of squared model parameters and weigh this term by alpha/2 * (1 / batch_size).
        Ensure that you return a Scalar object since we need to backpropagate through this term.
        """
        #if alpha = 0 we add a term 0, not the most elegant routine but it works
        if self.alpha == 0:
            return Scalar(0.0)

        squared_sum = Scalar(0.0)
        for p in self.model.parameters():
            squared_sum = squared_sum + (p * p)
        return squared_sum*self.alpha/2*(1/self.batch_size)

    def sgd_step(self, learning_rate: float) -> None:
        """
        Perform one step of stochastic gradient descent.
        Gradients are expected to be already calculated and available in p.grad if p is a parameter of self.model.
        """
        for p in self.model.parameters():
            p.value -= learning_rate * p.grad

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the given data.

        :param X: Features
        :param y: Targets
        """
        self.num_classes = len(set(y))
        assert self.num_classes > 1, "Number of classes must be greater than 1"
        if self.num_classes == 2:
            nn_num_outputs = 1
            '''
            raise NotImplementedError(
                "Task 3 (Binary classification) is not implemented. "
                "Thus, number of classes must be greater than 2 (multi-class classification)"
            )
            '''
        else:
            nn_num_outputs = self.num_classes

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        random_idxs = np.random.permutation(X.shape[0])
        num_batches = (
            X.shape[0] // self.batch_size
        )  # We will just skip the last batch if it's not a full batch

        self.model = MultiLayerPerceptron(
            num_inputs=X.shape[1],
            num_hidden=list(self.hidden_layer_sizes),
            num_outputs=nn_num_outputs,
        )

        for epoch_idx in range(self.num_epochs):
            learning_rate = 1.0 - 0.9 * epoch_idx / self.num_epochs
            for batch_idx in range(num_batches):
                idxs_in_batch = random_idxs[
                    self.batch_size * batch_idx : self.batch_size * (batch_idx + 1)
                ]
                X_batch, y_batch = X[idxs_in_batch], y[idxs_in_batch]

                losses, is_pred_correct = [], []
                for xi, yi in zip(X_batch, y_batch):
                    out = self.model([Scalar(x) for x in xi])
                    if self.num_classes == 2:
                        prob = self.sigmoid(out[0])
                        sample_loss = self.binary_cross_entropy_loss(yi, prob)
                        is_pred_correct.append((prob.value >= 0.5) == yi)
                    else:
                        probs = self.softmax(out)
                        sample_loss = self.multiclass_cross_entropy_loss(yi, probs)
                        is_pred_correct.append(
                            np.argmax([p.value for p in probs]) == yi
                        )

                    losses.append(sample_loss)

                accuracy = np.mean(is_pred_correct)
                self.model.zero_grad()
                loss = sum(losses) / len(losses)
                if self.alpha > 0:
                    loss += self.l2_regularization_term()
                loss.backward()
                self.sgd_step(learning_rate)

                print(
                    f"Epoch {epoch_idx+1} | Batch {batch_idx+1}/{len(random_idxs) // self.batch_size} "
                    f"| Batch-Loss {loss.value:.4f} | Batch-Accuracy {accuracy * 100}%"
                )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Returns the accuracy of the model on the given data.

        :param X: Features
        :param y: Targets
        """
        is_pred_correct = []
        for xi, yi in zip(X, y):
            out = self.model([Scalar(x) for x in xi])
            if self.num_classes == 2:
                prob = self.sigmoid(out[0])
                is_pred_correct.append((prob.value >= 0.5) == yi)
            else:
                probs = self.softmax(out)
                is_pred_correct.append(np.argmax([p.value for p in probs]) == yi)

        return np.mean(is_pred_correct)
