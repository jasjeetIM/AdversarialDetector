import tensorflow as tf
import keras.backend as K
from keras.utils import np_utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def avg_l2_dist(orig, adv):
    """Get the mean l2 distortion between two orig and adv images"""
    l2_dist = 0.0
    for i in range(orig.shape[0]):
        l2_dist+= np.linalg.norm(orig[i] - adv[i])
    return l2_dist/orig.shape[0]


def visualize(image):
    """Visualize gray scale or color image"""
    plt.figure(figsize=(1, 1))
    if image.shape[-1] == 1:
       # image is in black and white
       image = image[:, :, 0]
       plt.imshow(image, cmap='Greys')
    else:
        # image is in color
        plt.imshow(image)
    plt.axis('off')
    plt.show()

#Normalize rows of a given matrix
def normalize(matrix):
    """Normalize each row vector in a matrix"""
    matrix_nm = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        matrix_nm[i] = matrix[i]/np.linalg.norm(matrix[i]) 
    return matrix_nm


def tf_print(tensor, transform=None):

    # Insert a custom python operation into the graph that does nothing but print a tensors value 
    def print_tensor(x):
        # x is typically a numpy array here so you could do anything you want with it,
        # but adding a transformation of some kind usually makes the output more digestible
        print(x if transform is None else transform(x))
        return x
    
    log_op = tf.py_func(print_tensor, [tensor], [tensor.dtype])[0]
    with tf.control_dependencies([log_op]):
        res = tf.identity(tensor)

    # Return the given tensor
    return res



def jsma_symbolic(x, y_target, model, theta=1, gamma=.05, clip_min=0.0, clip_max=1.0):
    """
    TensorFlow implementation of the JSMA (see https://arxiv.org/abs/1511.07528
    for details about the algorithm design choices).
    :param x: the input placeholder
    :param y_target: the target tensor
    :param model: keras model
    :param theta: delta for each feature adjustment
    :param gamma: a float between 0 - 1 indicating the maximum distortion
        percentage
    :param clip_min: minimum value for components of the example returned
    :param clip_max: maximum value for components of the example returned
    :return: a tensor for the adversarial example
    """

    nb_classes = int(y_target.shape[-1].value)
    nb_features = int(np.product(x.shape[1:]).value)
    print (nb_classes, nb_features)
    max_iters = 1#np.floor(nb_features * gamma / 2)
    print (max_iters)
    increase = bool(theta > 0)

    #tmp = np.ones((nb_features, nb_features), int)
    #np.fill_diagonal(tmp, 0)
    #zero_diagonal = tf.constant(tmp, tf.float32)

    # Compute our initial search domain. We optimize the initial search domain
    # by removing all features that are already at their maximum values (if
    # increasing input features---otherwise, at their minimum value).
    if increase:
        search_domain = tf.reshape(
                            tf.cast(x < clip_max, tf.float32),
                            [-1, nb_features])
    else:
        search_domain = tf.reshape(
                            tf.cast(x > clip_min, tf.float32),
                            [-1, nb_features])
                               
    # Loop variables
    # x_in: the tensor that holds the latest adversarial outputs that are in
    #       progress.
    # y_in: the tensor for target labels
    # domain_in: the tensor that holds the latest search domain
    # cond_in: the boolean tensor to show if more iteration is needed for
    #          generating adversarial samples
    def condition(x_in, y_in, domain_in, i_in, cond_in, max_it):
        # Repeat the loop until we have achieved misclassification or
        # reaches the maximum iterations
        tf_print(i_in)
        return tf.logical_and(tf.less(i_in, max_it), cond_in)

    # Same loop variables as above
    def body(x_in, y_in, domain_in, i_in, cond_in, max_it):
        tf_print(max_it)
        preds = model(x_in)
        preds = tf_print(preds)
        preds_onehot = tf.one_hot(tf.argmax(preds, axis=1), depth=nb_classes)

        # create the Jacobian graph
        list_derivatives = []
        for class_ind in xrange(nb_classes):
            derivatives = tf.gradients(preds[:, class_ind], x_in)
            list_derivatives.append(derivatives[0])
        grads = tf.reshape(tf.stack(list_derivatives),
                           shape=[nb_classes, -1, nb_features])

        # Compute the Jacobian components
        # To help with the computation later, reshape the target_class
        # and other_class to [nb_classes, -1, 1].
        # The last dimention is added to allow broadcasting later.
        target_class = tf.reshape(tf.transpose(y_in, perm=[1, 0]),
                                  shape=[nb_classes, -1, 1])
        other_classes = tf.cast(tf.not_equal(target_class, 1), tf.float32)

        grads_target = tf.reduce_sum(grads * target_class, axis=0)
        grads_other = tf.reduce_sum(grads * other_classes, axis=0)

        # Remove the already-used input features from the search space
        # Subtract 2 times the maximum value from those values so that
        # they won't be picked later
        increase_coef = (4 * int(increase) - 2) \
            * tf.cast(tf.equal(domain_in, 0), tf.float32)

        target_tmp = grads_target
        target_tmp -= increase_coef \
            * tf.reduce_max(tf.abs(grads_target), axis=1, keep_dims=True)
        #target_sum = tf.reshape(target_tmp, shape=[-1, nb_features, 1]) \
        #    + t[f.reshape(target_tmp, shape=[-1, 1, nb_features])]

        other_tmp = grads_other
        other_tmp += increase_coef \
            * tf.reduce_max(tf.abs(grads_other), axis=1, keep_dims=True)
        #other_sum = tf.reshape(other_tmp, shape=[-1, nb_features, 1]) \
        #    + tf.reshape(other_tmp, shape=[-1, 1, nb_features])

        # Create a mask to only keep features that match conditions
        if increase:
            scores_mask = ((target_tmp > 0) & (other_tmp < 0))
        else:
            scores_mask = ((target_tmp < 0) & (other_tmp > 0))

        
        scores = tf.cast(scores_mask, tf.float32) \
            * (-target_tmp * other_tmp)# * zero_diagonal
        #scores = tf_print(scores)
        # Extract the best pixel
        best = tf.argmax(
                    tf.reshape(scores, shape=[-1, nb_features]),
                    axis=1)

        p1 = tf.mod(best, nb_features)
        #p1 = tf_print(p1)
       # p2 = tf.floordiv(best, nb_features)
        p1_one_hot = tf.one_hot(p1, depth=nb_features)
        #p2_one_hot = tf.one_hot(p2, depth=nb_features)

        # Check if more modification is needed for each sample
        mod_not_done = tf.equal(tf.reduce_sum(y_in * preds_onehot, axis=1), 0)
        cond = mod_not_done & (tf.reduce_sum(domain_in, axis=1) >= 1)
        #cond = tf_print(cond)
        # Update the search domain
        cond_float = tf.reshape(tf.cast(cond, tf.float32), shape=[-1, 1])
        to_mod = (p1_one_hot) * cond_float

        domain_out = domain_in - to_mod

        # Apply the modification to the images
        to_mod_reshape = tf.reshape(to_mod,
                                    shape=([-1] + x_in.shape[1:].as_list()))
        if increase:
            x_out = tf.minimum(clip_max, x_in + to_mod_reshape * theta)
        else:
            x_out = tf.maximum(clip_min, x_in - to_mod_reshape * theta)
        test_pred = model(x_out)
        #test_preds = tf_print(test_pred)
        # Increase the iterator, and check if all misclassifications are done
        i_out = tf.add(i_in, 1)
        #cond = tf_print(cond)
        #bb = tf.reduce_any(cond)
        cond_out = tf.reduce_any(cond)
        #bb = tf_print(bb)

        return x_out, y_in, domain_out, i_out, cond_out, max_it

    # Run loop to do JSMA
    print ('Starting loop')
    x_adv, y_, dom_, i_, cond_ ,max_ = tf.while_loop(condition, body, [x, y_target, search_domain, 1, True, max_iters], parallel_iterations=1)
    print ('done loop')
    return x_adv, y_, dom_, i_, cond_ ,max_

