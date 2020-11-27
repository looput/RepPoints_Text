from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def smooth_l1loss(target_tensor,prediction_tensor,delta=1.,weights=None):
    loss = tf.reduce_sum(tf.losses.huber_loss(
        target_tensor,
        prediction_tensor,
        delta=delta,
        weights=weights,
        loss_collection=None,
        reduction=tf.losses.Reduction.NONE
    ), axis=2)

    return loss

def sigmoid_focalloss(prediction_tensor,
                    target_tensor,
                    weights=None,
                    gamma=2.0, alpha=0.25):
    """Compute loss function.
    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape, either [batch_size, num_anchors,
        num_classes] or [batch_size, num_anchors, 1]. If the shape is
        [batch_size, num_anchors, 1], all the classses are equally weighted.
      class_indices: (Optional) A 1-D integer tensor of class indices.
        If provided, computes loss only for the specified class indices.
    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """
    # if class_indices is not None:
    #   weights *= tf.reshape(
    #       ops.indices_to_dense_vector(class_indices,
    #                                   tf.shape(prediction_tensor)[2]),
    #       [1, 1, -1])
    per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_tensor, logits=prediction_tensor))
    prediction_probabilities = tf.sigmoid(prediction_tensor)
    p_t = ((target_tensor * prediction_probabilities) +
           ((1 - target_tensor) * (1 - prediction_probabilities)))
    modulating_factor = 1.0
    if gamma:
      modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = 1.0
    if alpha is not None:
      alpha_weight_factor = (target_tensor * alpha +
                             (1 - target_tensor) * (1 - alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                per_entry_cross_ent)
    if weights is not None:
        return focal_cross_entropy_loss * weights
    else:
        return focal_cross_entropy_loss