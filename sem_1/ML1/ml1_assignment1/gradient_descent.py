import numpy as np
from typing import Callable, Tuple


def gradient_descent(f: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                     df: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                     x0: np.ndarray, 
                     y0: np.ndarray, 
                     learning_rate: float, 
                     lr_decay: float, 
                     num_iters: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find a local minimum of the function f(x, y) using gradient descent:
    Until the number of iteration is reached, decrease the current x and y points by the
    respective partial derivative times the learning_rate.
    In each iteration, record the current function value in the list f_list 
    and the current x and y points in the lists x_list and y_list.
    The function should return the lists x_list, y_list, f_list.

    :param f: Function to minimize
    :param df: Gradient of f i.e, function that computes gradients
    :param x0: initial x0 point
    :param y0: initial y0 point
    :param learning_rate: Learning rate
    :param lr_decay: Learning rate decay
    :param num_iters: number of iterations
    :return: x_list, y_list, f_list (lists of x, y, and f values over iterations). 
             The first element of the lists represents the initial point (and the function value at the initial point).
             The last element of the lists represents the final point (and the function value at the final point).
    """
    f_list = np.zeros(num_iters+1)
    x_list = np.zeros(num_iters+1)
    y_list = np.zeros(num_iters+1)
    f_list[0] = f(x0, y0)
    x_list[0] = x0
    y_list[0] = y0
    #choose starting point
    point_2d = np.array([x_list[0], y_list[0]])
    for iter in range(num_iters):
        #decay learning rate at each time step
        #alternative methods of lr decay are after being stuck for a certain number of epochs (eg 10)
        #or after a predefined number of epochs in general (eg. every 1000)
        point_2d -= learning_rate/(1 + lr_decay * iter) * df(x_list[iter], y_list[iter])
        x = point_2d[0]
        y = point_2d[1]
        x_list[iter+1] = x
        y_list[iter+1] = y
        f_list[iter+1] = f(x, y)
    return x_list, y_list, f_list


def griewank(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Implements the Griewank function (as specified in the assignment sheet)
    :param x: x-coordinate
    :param y: y-coordinate
    :return: Griewank function value
    """
    # TODO: Implement the Griewank function (as specified in the Assignment 1 sheet)
    f = 1 +(x**2 + y**2)/4000 - np.cos(x)*np.cos(y/np.sqrt(2))
    return f


def gradient_griewank(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Implements partial derivatives of Griewank function w.r.t. x and y
    :param x: x-coordinate
    :param y: y-coordinate
    :return: Gradient of Griewank function
    """
    # TODO: Implement partial derivatives of the Griewank function w.r.t. x and y
    df_dx = x / 2000 + np.sin(x)*np.cos(y/np.sqrt(2))
    df_dy = y / 2000 + np.sin(y/np.sqrt(2))*np.cos(x)/np.sqrt(2)

    gradient = np.array([df_dx, df_dy])
    return gradient


def finite_difference_gradient_approx(f: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                                      x: np.ndarray, 
                                      y: np.ndarray, 
                                      h: float = 1e-5) -> np.ndarray:
    """
    Implement finite difference gradient approximation.
    :param f: Function to approximate the gradient of
    :param x: x-coordinate
    :param y: y-coordinate
    :param h: Step size
    :return: Approximated gradient
    """
    # TODO: Implement numerical approximation to the partial derivatives of 
    # the Griewank function w.r.t. x and y
    df_dx = (f(x+h, y) - f(x-h, y))/(2*h)
    df_dy = (f(x, y+h) - f(x, y-h))/(2*h)

    approx_grad = np.array([df_dx, df_dy])
    return approx_grad