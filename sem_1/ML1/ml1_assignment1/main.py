import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from matplotlib.pyplot import title
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from plot_utils import (plot_scatterplot_and_line, plot_scatterplot_and_polynomial,
                        plot_logistic_regression, plot_datapoints, plot_3d_surface, plot_2d_contour,
                        plot_function_over_iterations)
from linear_regression import (fit_univariate_lin_model,
                               fit_multiple_lin_model,
                               univariate_loss, multiple_loss,
                               calculate_pearson_correlation,
                               compute_design_matrix,
                               compute_polynomial_design_matrix)
from logistic_regression import (create_design_matrix_dataset_1,
                                 create_design_matrix_dataset_2,
                                 create_design_matrix_dataset_3,
                                 logistic_regression_params_sklearn)
from gradient_descent import griewank, gradient_griewank, gradient_descent, finite_difference_gradient_approx


def task_1(use_linalg_formulation=False):
    print('---- Task 1 ----')
    column_to_id = {"mpg": 0, "cylinders": 1,
                    "displacement": 2, "horsepower": 3,
                    "weight": 4, "acceleration": 5,
                    "model_year": 6, "origin": 7}

    # After loading the data, you can for example access it like this:
    # `cars_data[:, column_to_id['weight']]`
    cars_data = np.load('data/cars_data.npy')
    # TODO: Implement Task 1.1.2: Find 3 pairs of features that have a linear relationship.

    lin_relations = np.array([('horsepower', 'weight'), ('displacement', 'horsepower'), ('horsepower', 'acceleration')])
    # get data values to match chosen pairs of linear relations
    data_lin_rel = np.array(
        [(cars_data[:, column_to_id[f'{col_name_1}']], cars_data[:, column_to_id[f'{col_name_2}']]) for
         col_name_1, col_name_2 in lin_relations])

    non_lin_relations = np.array([('cylinders', 'weight'), ('model_year', 'origin'), ('cylinders', 'horsepower')])
    data_non_lin_rel = np.array(
        [(cars_data[:, column_to_id[f'{col_name_1}']], cars_data[:, column_to_id[f'{col_name_2}']]) for
         col_name_1, col_name_2 in non_lin_relations])
    # For each pair, fit a univariate linear regression model: If ``use_linalg_formulation`` is False,
    # call `fit_univariate_lin_model`, otherwise use the linalg formulation in `fit_multiple_lin_model` (Task 1.2.2).
    # For each pair, also calculate and report the Pearson correlation coefficient, the theta vector you found,
    # the MSE, and plot the data points together with the linear function.
    for (names, data) in zip(lin_relations, data_lin_rel):
        xname, yname = names
        x, y = data
        if use_linalg_formulation:
            X_design = compute_design_matrix(x.reshape(-1, 1))
            theta = fit_multiple_lin_model(X_design, y)
        else:
            theta = fit_univariate_lin_model(x, y)
        pearson = calculate_pearson_correlation(x, y)
        MSE = univariate_loss(x, y, theta)
        plot_scatterplot_and_line(x, y, theta, xlabel=xname,
                                  ylabel=yname,
                                  title=f'linfit {yname} over {xname}. Params to 3 degrees: \n'
                                        f'theta = ({theta[0]:.3f}, {theta[1]:.3f}), pearson = {pearson:.3f}, MSE = {MSE:.3f}',
                                  figname = f'{xname}_{yname}_linfit')

        # Repeat the process for 3 pairs of features that do not have a meaningful linear relationship.
    for (names, data) in zip(non_lin_relations, data_non_lin_rel):
        xname, yname = names
        x, y = data
        if use_linalg_formulation:
            X_design = compute_design_matrix(x.reshape(-1, 1))
            theta = fit_multiple_lin_model(X_design, y)
        else:
            theta = fit_univariate_lin_model(x, y)
        pearson = calculate_pearson_correlation(x, y)
        MSE = univariate_loss(x, y, theta)
        plot_scatterplot_and_line(x, y, theta, xlabel=xname,
                                  ylabel=yname,
                                  title=f'linfit {yname} over {xname}. Params to 3 degrees: \n'
                                        f'theta = ({theta[0]:.3f}, {theta[1]:.3f}), pearson = {pearson:.3f}, MSE = {MSE:.3f}',
                                  figname=f'{xname}_{yname}_non_lin_linfit.pdf')

    # TODO: Implement Task 1.2.3: Multiple linear regression
    # hardcoded three features and one output, with one feature and the output being present in task 1 as well
    x1_name = 'displacement'
    x2_name = 'acceleration'
    x3_name = 'weight'

    x1 = cars_data[:, column_to_id[x1_name]]
    x2 = cars_data[:, column_to_id[x2_name]]
    x3 = cars_data[:, column_to_id[x3_name]]

    features_t2 = np.array([x1, x2, x3]).T
    X_t2 = compute_design_matrix(features_t2)

    y_name_t2 = 'horsepower'

    y_t2 = cars_data[:, column_to_id[y_name_t2]]

    theta_t2 = fit_multiple_lin_model(X_t2, y_t2)
    L_M_t2 = multiple_loss(X_t2, y_t2, theta_t2)
    print(f'L2-regularization loss with three features: {L_M_t2:.3f}. Theta: {theta_t2}')
    # Select two additional features, compute the design matrix, and fit the multiple linear regression model.
    # Report the MSE and the theta vector.

    # TODO: Implement Task 1.3.2: Polynomial regression
    # redeclaring functions. technically pointelss here, but in general the necessary data isn't necessarily predefined
    x4_name = 'displacement'
    y_name_t3 = 'horsepower'

    x4 = cars_data[:, column_to_id[x4_name]]
    y_t3 = cars_data[:, column_to_id[y_name_t3]]

    K = 4
    X_K = compute_polynomial_design_matrix(x4, K)
    theta_k = fit_multiple_lin_model(X_K, y_t3)
    L_M_t3 = multiple_loss(X_K, y_t3, theta_k)
    plot_scatterplot_and_polynomial(x4, y_t3, theta_k, xlabel=x4_name, ylabel=y_name_t3,
                                    title=f'polynomial fit of degree {K} for {y_name_t3} over {x4_name}.\n'
                                          f'max component theta = {np.max(theta_k)}'
                                          f' Loss: {L_M_t3:.3f}', figname=f'{x4_name}_{y_name_t3}_polyfit.pdf')
    # For the feature-target pair of choice, compute the polynomial design matrix with an appropriate degree K,
    # fit the model, and plot the data points together with the polynomial function.
    # Report the MSE and the theta vector.
    print(f'task 1 polynomial regression, MSE: {L_M_t3},\n theta: {theta_k}')

    # TODO: Implement Task 1.3.3: Use x_small_1 and y_small_1 to fit a polynomial model.
    # Find and report the smallest K that gets zero loss. Plot the data points and the polynomial function.
    x_small_1 = cars_data[10:15, column_to_id['displacement']]
    y_small_1 = cars_data[10:15, column_to_id['weight']]

    K_3 = 4
    X_K_3 = compute_polynomial_design_matrix(x_small_1, K_3)
    #get perfect inverse here, to get perfect fit:
    def fit_model_with_zero_loss(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        theta, a, b, c, = np.linalg.lstsq(X_K_3, y, rcond=None)
        return theta
    #we do not get loss zero here with the linalg pinv function, since it is only a pseudoinverse
    #and fails in case of duplicate values or numerical problems
    #hence, the line below is commented out, to show the theoretical behaviour with true inversion.
    # theta_k_3 = fit_model_with_zero_loss(X_K_3, y_small_1)
    #true inversion used, to calculate theta perfectly:
    theta_k_3 = fit_multiple_lin_model(X_K_3, y_small_1)

    L_M_small_1 = multiple_loss(X_K_3, y_small_1, theta_k_3)
    plot_scatterplot_and_polynomial(x_small_1, y_small_1, theta_k_3, xlabel='displacement', ylabel='weight',
                                    title=f'polynomial fit of degree {K_3} for displacement over weight. \n'
                                          f' small set 1, Loss: {L_M_small_1:.3f}', figname='weight_displacement_small_polyfit.pdf')


    # TODO: Implement 1.3.4: Use the K from Task 1.3.3, x_small_2 and y_small_2 to fit a polynomial model.
    # Plot the data points and the polynomial function. Report why the loss is not zero in this case.

    x_small_2 = cars_data[169:174, column_to_id['displacement']]
    y_small_2 = cars_data[169:174, column_to_id['weight']]
    X_K_4 = compute_polynomial_design_matrix(x_small_2, K_3)
    #show failure of pinv
    theta_k_4 = fit_multiple_lin_model(X_K_4, y_small_2)
    L_M_small_2 = multiple_loss(X_K_4, y_small_2, theta_k_4)
    plot_scatterplot_and_polynomial(x_small_1, y_small_1, theta_k_3, xlabel='displacement', ylabel='weight',
                                    title=f'polynomial fit of degree {K_3} for displacement over weight. \n'
                                          f' small set 2, Loss: {L_M_small_2:.3f}', figname='weight_displacement_small_2_polyfit.pdf')




def task_2():
    print('\n---- Task 2 ----')

    for task in [1, 2, 3]:
        print(f'---- Logistic regression task {task} ----')
        if task == 1:
            # TODO: Load the data set 1 (X-1-data.npy and targets-dataset-1.npy)
            X_data = np.load('data/X-1-data.npy')
            y = np.load('data/targets-dataset-1.npy')
            create_design_matrix = create_design_matrix_dataset_1
        elif task == 2:
            # TODO: Load the data set 2 (X-2-data.npy and targets-dataset-2.npy)
            X_data = np.load('data/X-2-data.npy')
            y = np.load('data/targets-dataset-2.npy')
            create_design_matrix = create_design_matrix_dataset_2
        elif task == 3:
            # TODO: Load the data set 3 (X-3-data.npy and targets-dataset-3.npy)
            X_data = np.load('data/X-3-data.npy')
            y = np.load('data/targets-dataset-3.npy')
            create_design_matrix = create_design_matrix_dataset_3
        else:
            raise ValueError('Task not found.')

        X = create_design_matrix(X_data)

        # Plot the datapoints (just for visual inspection)
        plot_datapoints(X, y, f'Targets - Task {task}')

        # TODO: Split the dataset using the `_split` function.
        # The parameter `random_state` should be set to 0.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        print(
            f'Shapes of: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}')

        # Train the classifier
        custom_params = logistic_regression_params_sklearn()
        clf = LogisticRegression(**custom_params)
        # TODO: Fit the model to the data using the `fit` method of the classifier `clf`
        clf.fit(X_train, y_train)
        # TODO: Use the `score` method of the classifier `clf` to calculate accuracy
        acc_train, acc_test = clf.score(X_train, y_train), clf.score(X_test, y_test)

        print(f'Train accuracy: {acc_train * 100:.3f}%. Test accuracy: {100 * acc_test:.3f}%.')

        # TODO: Use the `predict_proba` method of the classifier `clf` to
        #  calculate the predicted probabilities on the training set
        yhat_train = clf.predict_proba(X_train)
        yhat_test = clf.predict_proba(X_test)
        # TODO: Use the `predict_proba` method of the classifier `clf` to
        #  calculate the predicted probabilities on the test set

        # TODO: Use the `log_loss` function to calculate the cross-entropy loss
        #  (once on the training set, once on the test set).
        #  You need to pass (1) the true binary labels and (2) the probability of the *positive* class to `log_loss`.
        #  Since the output of `predict_proba` is of shape (n_samples, n_classes), you need to select the probabilities
        #  of the positive class by indexing the second column (index 1).
        loss_train, loss_test = log_loss(y_train, yhat_train), log_loss(y_test, yhat_test)
        print(f'Train loss: {loss_train}. Test loss: {loss_test}.')

        plot_logistic_regression(clf, create_design_matrix, X_train, f'(Dataset {task}) Train set predictions',
                                 figname=f'logreg_train{task}')
        plot_logistic_regression(clf, create_design_matrix, X_test, f'(Dataset {task}) Test set predictions',
                                 figname=f'logreg_test{task}')

        # TODO: Print theta vector (and also the bias term). Hint: Check the attributes of the classifier
        classifier_weights, classifier_bias = clf.coef_[0], clf.intercept_[0]
        print(f'Parameters: {classifier_weights}, {classifier_bias}')


def task_3(initial_plot=True):
    print('\n---- Task 3 ----')
    # Do *not* change this seed
    np.random.seed(0)

    # TODO: Choose a random starting point using samples from a uniform distribution
    x0 = np.random.uniform(-10,10)
    y0 = np.random.uniform(-10,10)
    print(f'Starting point: {x0:.4f}, {y0:.4f}')

    if initial_plot:
        # Plot the function to see how it looks like
        plot_3d_surface(griewank)
        plot_2d_contour(griewank, starting_point=(x0, y0), global_min=(0, 0))

    # TODO: Check if gradient_griewank is correct at (x0, y0).
    # To do this, print the true gradient and the numerical approximation.
    true_grad = gradient_griewank(x0, y0)
    num_approx_grad = finite_difference_gradient_approx(griewank, x0, y0)
    print('Comparison of true gradient with finite difference gradient: \n',
          f'True gradient: {true_grad[0]}, {true_grad[1]},\n'
          f' finite difference gradient: {num_approx_grad[0]}, {num_approx_grad[1]}')

    # TODO: Call the function `gradient_descent` with a chosen configuration of hyperparameters,
    #  i.e., learning_rate, lr_decay, and num_iters. Try out lr_decay=1 as well as values for lr_decay that are < 1.
    x_list, y_list, f_list = gradient_descent(griewank, gradient_griewank, x0, y0, 0.001, 1e-5, 10000)

    # Print the point that is found after `num_iters` iterations
    print(f'Solution found: f({x_list[-1]:.4f}, {y_list[-1]:.4f})= {f_list[-1]:.4f}')
    print(f'Global optimum: f(0, 0)= {griewank(0, 0):.4f}')

    # Here we plot the contour of the function with the path taken by the gradient descent algorithm
    plot_2d_contour(griewank, starting_point=(x0, y0), global_min=(0, 0),
                    x_list=x_list, y_list=y_list)

    # TODO: Create a plot f(x_t, y_t) over iterations t by calling `plot_function_over_iterations` with `f_list`
    plot_function_over_iterations(f_list, 'gradient_descent_function_values_over_iterations')
    pass


def main():
    np.random.seed(46)

    task_1(use_linalg_formulation=False)
    task_2()
    task_3(initial_plot=True)


if __name__ == '__main__':
    main()
