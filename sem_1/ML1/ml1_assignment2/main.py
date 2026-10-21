import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nn_classification_from_scratch import train_nn_own
from nn_classification_sklearn import (
    reduce_dimension,
    train_nn,
    train_nn_with_regularization,
    perform_grid_search,
    show_confusion_matrix_and_classification_report,
    plot_training_loss_curve,
)


def task_1():
    features = np.load("data/chest_xray_images.npy")
    targets = np.load("data/chest_xray_targets.npy")
    features = features.reshape((features.shape[0], -1))
    print(features.shape, targets.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    # PCA
    # Task 1.1.1
    print("----- Task 1.1.1 -----")
    X_train_pca, pca = reduce_dimension(X_train, n_components=160)

    # Task 1.1.2
    print("----- Task 1.1.2 -----")
    best_nn = train_nn(X_train_pca, y_train)

    # Task 1.1.3 needs no code, only explanations in the report

    # Task 1.1.4
    print("----- Task 1.1.4 -----")
    best_reg_nn = train_nn_with_regularization(X_train_pca, y_train)

    best_model_task_1_1 = best_reg_nn  # TODO: Choose the best model from the previous Tasks
    plot_training_loss_curve(best_model_task_1_1)

    # Task 1.2
    print("----- Task 1.2 -----")
    best_gs_nn = perform_grid_search(X_train_pca, y_train)

    final_model = best_gs_nn  # TODO: Choose the best model from *all* previous Tasks

    X_test_pca = pca.transform(X_test)
    show_confusion_matrix_and_classification_report(final_model, X_test_pca, y_test)


def task_2():
    features = np.load("data/chest_xray_images.npy")
    targets = np.load("data/chest_xray_targets.npy")
    features = features.reshape((features.shape[0], -1))

    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    X_train_pca, pca = reduce_dimension(
        X_train, n_components=16
    )  # Let's project the features to 16 dimensions

    # We scale the features mainly because our implementation is not numerically stable
    scaler = StandardScaler()
    X_train_pca = scaler.fit_transform(X_train_pca)

    nn = train_nn_own(X_train_pca, y_train)
    # alpha 0-> training accuracy 0.7526041666666666 and validation accuracy 0.6848958333333334
    #alpha 0.01 -> training accuracy 0.7506510416666666 and validation accuracy 0.6901041666666666
    # alpha = 1.0 -> training accuracy 0.5735677083333334 and validation accuracy 0.6015625

    X_test_pca = pca.transform(X_test)
    X_test_pca = scaler.transform(X_test_pca)

    test_acc = nn.score(X_test_pca, y_test)
    print(f"Test accuracy: {test_acc:.4f}.")


def task_3():
    features = np.load("data/chest_xray_images.npy")
    targets = np.load("data/chest_xray_targets.npy")
    features = features.reshape((features.shape[0], -1))

    # Create a binary classification dataset out of the multi-class dataset
    # (by keeping only the samples with targets 0 and 1)
    idxs = np.where(targets < 2)[0]
    features = features[idxs]
    targets = targets[idxs]

    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    X_train_pca, pca = reduce_dimension(
        X_train, n_components=16
    )  # Let's project the features to 16 dimensions

    # We scale the features mainly because our implementation is not numerically stable
    scaler = StandardScaler()
    X_train_pca = scaler.fit_transform(X_train_pca)

    nn = train_nn_own(X_train_pca, y_train)

    X_test_pca = pca.transform(X_test)
    X_test_pca = scaler.transform(X_test_pca)

    test_acc = nn.score(X_test_pca, y_test)
    print(f"Test accuracy: {test_acc:.4f}.")


def main():
    task_1()
    task_2()
    task_3()


if __name__ == "__main__":
    main()
