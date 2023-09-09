import paddle

def dot_product(a, b):
    """
    Compute the dot product between two tensors using PaddlePaddle.

    Args:
        a (paddle.Tensor): The first input tensor.
        b (paddle.Tensor): The second input tensor.

    Returns:
        paddle.Tensor: The dot product of the two input tensors.
    """
    return paddle.sum(a * b)


if __name__ == "__main__":
    a = paddle.to_tensor([1, 2, 3])
    b = paddle.to_tensor([4, 5, 6])

    result = dot_product(a, b)
    print("Dot Product:", result.numpy())
