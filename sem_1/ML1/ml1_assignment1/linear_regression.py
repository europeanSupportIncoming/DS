import numpy as np


def univariate_loss(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    :param x: 1D array that represents the feature vector
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector theta = (b, w)
    :return: a scalar that represents the loss \mathcal{L}_U(theta)
    """
    N = x.size
    assert N == y.size, "output dim must match target dim"
    f_theta_x = np.ones_like(x)*theta[0] + theta[1] * x
    Loss = np.sum((f_theta_x - y)**2) /N
    return Loss


def fit_univariate_lin_model(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param x: 1D array that contains the feature of each subject
    :param y: 1D array that contains the target of each subject
    :return: the parameter vector theta^* that minimizes the loss \mathcal{L}_U(theta)
    """

    N = x.size
    assert N > 1, "There must be at least 2 points given!"
    # TODO: Implement the expressions you have derived in the pen & paper exercise (Task 1.1.1)
    x_bar = np.ones_like(x)*np.sum(x)/N
    y_bar = np.ones_like(y)*np.sum(y)/N
    w = np.sum((x-x_bar)*(y-y_bar))/np.sum((x-x_bar)**2)
    #get 0th element, which is the average, instead of using an array of that
    #element in shape of our vector
    b = y_bar[0] - w * x_bar[0]
    return np.array([b, w])


def calculate_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: 1D array that contains the feature of each subject
    :param y: 1D array that contains the target of each subject
    :return: a scalar that represents the Pearson correlation coefficient between x and y
    """
    # TODO: Implement Pearson correlation coefficient, as shown in Equation 3 (Task 1.1.2).
    N = x.size
    x_bar = np.ones_like(x) * np.sum(x) / N
    y_bar = np.ones_like(y) * np.sum(y) / N
    pearson_r = np.sum((x-x_bar)*(y-y_bar))/np.sqrt(np.sum((x-x_bar)**2)*np.sum((y-y_bar)**2))
    return pearson_r


def compute_design_matrix(data: np.ndarray) -> np.ndarray:
    """
    :param data: 2D array of shape (N, D) that represents the data matrix
    :return: 2D array that represents the design matrix. Think about the shape of the output.
    """

    # TODO: Implement the design matrix for multiple linear regression (Task 1.2.2)
    N = len(data)
    D = len(data[0])
    design_matrix = np.ones((N, D+1))
    for i in range(N):
        for j in range(D):
            design_matrix[i, j+1] = data[i,j]
    return design_matrix


def multiple_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    :param X: 2D array that represents the design matrix
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector
    :return: a scalar that represents the loss \mathcal{L}_M(theta)
    """
    # TODO: Implement the multiple regression loss \mathcal{L}_M(theta) (as specified in Equation 5)
    N = len(y)
    y_hat = X.dot(theta)
    Loss = np.sum((y_hat - y)**2)/N
    return Loss


def fit_multiple_lin_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param X: 2D array that represents the design matrix
    :param y: 1D array that represents the target vector
    :return: the parameter vector theta^* that minimizes the loss \mathcal{L}_M(theta)
    """
    from numpy.linalg import pinv

    # TODO: Implement the expressions you have derived in the pen & paper exercise (Task 1.2.1). 
    # Note: Use the pinv function.

    MP_pinv_matrix = np.linalg.pinv(X.T@X)
    theta = MP_pinv_matrix.dot(X.T@y)
    return theta


def compute_polynomial_design_matrix(x: np.ndarray, K: int) -> np.ndarray:
    """
    :param x: 1D array that represents the feature vector
    :param K: the degree of the polynomial
    :return: 2D array that represents the design matrix. Think about the shape of the output.
    """

    # TODO: Implement the polynomial design matrix (Task 1.3.2)
    polynomial_design_matrix = np.array([x**k for k in range(K+1)]).T
    return polynomial_design_matrix