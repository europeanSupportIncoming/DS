from autodiff.neural_net import Neuron, FeedForwardLayer, MultiLayerPerceptron
from autodiff.scalar import Scalar


def neuron_test(use_relu: bool):
    """
    Test the Neuron class with a fixed set of weights and a bias and fixed input.
    """
    neuron = Neuron(3, use_relu=use_relu)
    neuron.w = [Scalar(1), Scalar(2), Scalar(3)]  # Let's set the weights to [1, 2, 3]
    neuron.b = Scalar(-1)  # Let's set the bias to -1

    out = neuron(
        [Scalar(-1), Scalar(2), Scalar(-3)]
    )  # Let's pass the input [-1, 2, -3] to the neuron
    # check if `out` is a Scalar object
    assert isinstance(out, Scalar), f"Expected a Scalar object, but got {type(out)}"
    expected_out = 1 * (-1) + 2 * 2 + 3 * (-3) - 1
    if use_relu:
        expected_out = max(0, expected_out)

    assert (
        out.value == expected_out
    ), f"Neuron output ({out.value}) does not match expected value ({expected_out})"


def neuron_gradient_test(use_relu: bool):
    """
    Test the gradients of the Neuron class with a fixed set of weights and a bias and fixed input.
    """
    neuron = Neuron(3, use_relu=use_relu)
    neuron.w = [Scalar(1), Scalar(2), Scalar(3)]  # Let's set the weights to [1, 2, 3]
    neuron.b = Scalar(-1)  # Let's set the bias to -1

    out = neuron(
        [Scalar(-1), Scalar(2), Scalar(-3)]
    )  # Let's pass the input [-1, 2, -3] to the neuron

    out.backward()  # Let's compute the gradient of `out` w.r.t. all involved Scalar objects

    # check if the gradients are correct
    expected_grads = [-1, 2, -3, 1] if not use_relu else [0, 0, 0, 0]
    calc_grads = [w.grad for w in neuron.w] + [neuron.b.grad]
    for calc_grad, expected_grad in zip(calc_grads, expected_grads):
        assert (
            calc_grad == expected_grad
        ), f"Gradient ({calc_grad}) does not match expected value ({expected_grad})"


def layer_test(use_relu):
    """
    Test the FeedForwardLayer class with a fixed set of weights and biases and fixed input.
    """
    layer = FeedForwardLayer(3, 2, use_relu=use_relu)
    layer.neurons[0].w = [
        Scalar(1),
        Scalar(2),
        Scalar(3),
    ]  # Let's set the weights of the first neuron to [1, 2, 3]
    layer.neurons[0].b = Scalar(-1)  # Let's set the bias of the first neuron to -1
    layer.neurons[1].w = [
        Scalar(-1),
        Scalar(-2),
        Scalar(-3),
    ]  # Let's set the weights of the second neuron to [-1, -2, -3]
    layer.neurons[1].b = Scalar(1)  # Let's set the bias of the second neuron to 1

    out = layer(
        [Scalar(-1), Scalar(2), Scalar(-3)]
    )  # Let's pass the input [-1, 2, -3] to the layer
    # check if `out` is a list of Scalar objects
    assert isinstance(
        out, list
    ), f"Expected a list of Scalar objects, but got {type(out)}"
    assert all(
        isinstance(o, Scalar) for o in out
    ), f"Expected a list of Scalar objects, but got {type(out[0])}"

    expected_out0 = 1 * (-1) + 2 * 2 + 3 * (-3) - 1
    expected_out1 = (-1) * (-1) + (-2) * 2 + (-3) * (-3) + 1
    if use_relu:
        expected_out0 = max(0, expected_out0)
        expected_out1 = max(0, expected_out1)

    assert (
        out[0].value == expected_out0
    ), f"Neuron 0 output ({out[0].value}) does not match expected value ({expected_out0})"
    assert (
        out[1].value == expected_out1
    ), f"Neuron 1 output ({out[1].value}) does not match expected value ({expected_out1})"


def mlp_test():
    """
    Test the MultiLayerPerceptron class with a fixed set of weights and biases and fixed input.
    """
    mlp = MultiLayerPerceptron(2, [3, 2], 1)
    mlp.layers[0].neurons[0].w = [Scalar(1), Scalar(2)]
    mlp.layers[0].neurons[0].b = Scalar(-1)
    mlp.layers[0].neurons[1].w = [Scalar(-1), Scalar(-2)]
    mlp.layers[0].neurons[1].b = Scalar(1)
    mlp.layers[0].neurons[2].w = [Scalar(0.5), Scalar(-0.5)]
    mlp.layers[0].neurons[2].b = Scalar(3)

    mlp.layers[1].neurons[0].w = [Scalar(1), Scalar(2), Scalar(3)]
    mlp.layers[1].neurons[0].b = Scalar(5)
    mlp.layers[1].neurons[1].w = [Scalar(-3), Scalar(-2), Scalar(-1)]
    mlp.layers[1].neurons[1].b = Scalar(-1)

    mlp.layers[2].neurons[0].w = [Scalar(-0.5), Scalar(1)]
    mlp.layers[2].neurons[0].b = Scalar(-1)

    out = mlp([Scalar(-1), Scalar(2)])  # Let's pass the input [-1, 2] to the MLP
    # check if `out` is a list of Scalar objects
    assert isinstance(
        out, list
    ), f"Expected a list of Scalar objects, but got {type(out)}"
    assert all(
        isinstance(o, Scalar) for o in out
    ), f"Expected a list of Scalar objects, but got {type(out[0])}"

    expected_hidden0_out0 = max(0, 1 * (-1) + 2 * 2 - 1)
    expected_hidden0_out1 = max(0, (-1) * (-1) + (-2) * 2 + 1)
    expected_hidden0_out2 = max(0, 0.5 * (-1) + (-0.5) * 2 + 3)

    expected_hidden1_out0 = max(
        0,
        1 * expected_hidden0_out0
        + 2 * expected_hidden0_out1
        + 3 * expected_hidden0_out2
        + 5,
    )
    expected_hidden1_out1 = max(
        0,
        (-3) * expected_hidden0_out0
        + (-2) * expected_hidden0_out1
        + (-1) * expected_hidden0_out2
        - 1,
    )
    expected_out = (
        (-0.5) * expected_hidden1_out0 + 1 * expected_hidden1_out1 - 1
    )  # Note: No ReLU at the output layer

    assert (
        out[0].value == expected_out
    ), f"MLP output ({out[0].value}) does not match expected value ({expected_out})"


if __name__ == "__main__":
    neuron_test(use_relu=False)
    neuron_test(use_relu=True)

    neuron_gradient_test(use_relu=False)
    neuron_gradient_test(use_relu=True)

    layer_test(use_relu=False)
    layer_test(use_relu=True)

    mlp_test()
