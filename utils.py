import tensorflow as tf


def l1_l2_regularizer(scale_l1=1.0, scale_l2=1.0, scope=None):
    """
    From Arimo DL SDK code
    Returns a function that can be used to apply L1 L2 regularizations.

    Args:
      scale_l1: A scalar multiplier `Tensor` for L1 regularization.
      scale_l2: A scalar multiplier `Tensor` for L2 regularization.
      scope: An optional scope name.

    Returns:
      A function with signature `l1_l2(weights)` that applies a weighted sum of
      L1 L2  regularization.

    Raises:
      ValueError: If scale is negative or if scale is not a float.

    Backport from TF master branch.
    """
    scope = scope or 'l1_l2_regularizer'
    return tf.contrib.layers.sum_regularizer([tf.contrib.layers.l1_regularizer(scale_l1),
                                              tf.contrib.layers.l2_regularizer(scale_l2)],
                                             scope=scope)
