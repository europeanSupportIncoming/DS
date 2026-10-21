from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import warnings

# We will suppress ConvergenceWarnings in this task. In practice, you should take warnings more seriously.
warnings.filterwarnings("ignore")


def reduce_dimension(X_train: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    :param X_train: Training data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality, which has shape (n_samples, n_components), and the PCA object
    """
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    reduced = pca.fit_transform(X_train)
    print(f'\nPCA explained variance ratio: {pca.explained_variance_ratio_}\n')
    return reduced, pca


def train_nn(X_train: np.ndarray, y_train: np.ndarray, layers = [(2, ), (8, ), (64, ), (256, ), (1024, ), (128, 256, 128)]) -> MLPClassifier:
    """
    Train MLPClassifier with different number of neurons and hidden layers.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    val_acs = np.zeros(len(layers))
    for it, hl in enumerate(layers):
        clf = MLPClassifier(solver='adam', max_iter=100, random_state=42, hidden_layer_sizes=hl).fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        val_ac = clf.score(X_val, y_val)
        val_acs[it] = val_ac + train_acc/100
        train_loss = clf.loss_
        print(f'\n layers: {hl} \nTrain accuracy: {train_acc}\n Validation accuracy: {val_ac}\n training loss: {train_loss}')
        # choice of best classifier: check which one gives highest validation accuracy
        # since that is most closely related to being able to predict new data
        # additionally, since it should be trained well, add 1% of training accuracy
    idx = np.argmax(val_acs)
    best_hl = layers[idx]
    clf = MLPClassifier(solver='adam', max_iter=100, random_state=42, hidden_layer_sizes=best_hl).fit(X_train, y_train)
    # TODO: Train MLPClassifier with different number of layers/neurons.
    #       Print the train accuracy, validation accuracy, and the training loss for each configuration.
    #       Return the MLPClassifier that you consider to be the best.
    return clf


def train_nn_with_regularization(
    X_train: np.ndarray, y_train: np.ndarray, layers = [(2, ), (8, ), (64, ), (256, ), (1024, ), (128, 256, 128)]
) -> MLPClassifier:
    """
    Train MLPClassifier using regularization.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    val_acs = np.zeros(6)
    for it, hl in enumerate(layers):
        clf = MLPClassifier(solver='adam', max_iter=100, random_state=42, hidden_layer_sizes=hl, alpha=1).fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        val_ac = clf.score(X_val, y_val)
        val_acs[it] = val_ac + train_acc/100
        train_loss = clf.loss_
        print(f'\n layers: {hl} \nTrain accuracy: {train_acc}\n Validation accuracy: {val_ac}\n training loss: {train_loss}')
        # choice of best classifier: check which one gives highest validation accuracy
        # since that is most closely related to being able to predict new data
        # additionally, since it should be trained well, add 1% of training accuracy
    idx = np.argmax(val_acs)
    best_hl = layers[idx]
    clf = MLPClassifier(solver='adam', max_iter=100, random_state=42, hidden_layer_sizes=best_hl).fit(X_train, y_train)
    # TODO: Use the code from the `train_nn` function, but add regularization to the MLPClassifier.
    #       Again, return the MLPClassifier that you consider to be the best.

    return clf



def plot_training_loss_curve(nn: MLPClassifier) -> None:
    """
    Plot the training loss curve.

    :param nn: The trained MLPClassifier
    """
    # TODO: Plot the training loss curve of the MLPClassifier. Don't forget to label the axes.
    loss_curve = nn.loss_curve_
    iters = len(loss_curve)
    xax = np.linspace(0, iters, iters)
    plt.scatter(xax, loss_curve, s=0.8)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()


def show_confusion_matrix_and_classification_report(
    nn: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray
) -> None:
    """
    Plot confusion matrix and print classification report.
    :param nn: The trained MLPClassifier you want to evaluate
    :param X_test: Test features (PCA-projected)
    :param y_test: Test targets
    """
    y_predict = nn.predict(X_test)
    nn_confusion_matrix = confusion_matrix(y_test, y_predict)
    disp = ConfusionMatrixDisplay(nn_confusion_matrix)
    disp.plot()
    plt.show()
    print(classification_report(y_test, y_predict))
    # TODO: Use `nn` to compute predictions on `X_test`.
    #       Use `confusion_matrix` and `ConfusionMatrixDisplay` to plot the confusion matrix on the test data.
    #       Use `classification_report` to print the classification report.


def perform_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Perform GridSearch using GridSearchCV.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The best estimator (MLPClassifier) found by GridSearchCV
    """
    parameters = {'alpha': [0.0, 0.1, 1.0], 'batch_size':[35, 256], 'hidden_layer_sizes':[(128,), (256,)]}
    mlp = MLPClassifier(solver='adam', max_iter=100, random_state=42)
    clf = GridSearchCV(mlp, parameters, cv=5, verbose=4)
    clf.fit(X_train, y_train)
    print(f'best score: {clf.best_score_} n best params: {clf.best_params_}')
    # TODO: Create parameter dictionary for GridSearchCV, as specified in the assignment sheet.
    #       Create an MLPClassifier with the specified default values.
    #       Run the grid search with `cv=5` and (optionally) `verbose=4`.
    #       Print the best score (mean cross validation score) and the best parameter set.
    #       Return the best estimator found by GridSearchCV.

    return clf.best_estimator_