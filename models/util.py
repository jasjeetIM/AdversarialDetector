import numpy as np
import matplotlib, gc
import matplotlib.pyplot as plt
from tensorflow import gradients
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops


def hessian_vector_product(ys, xs, v):
    """ Multiply the Hessian of `ys` wrt `xs` by `v` """ 

    # Validate the input
    length = len(xs)
    if len(v) != length:
         raise ValueError("xs and v must have the same length.")

    # First backprop
    grads = gradients(ys, xs)
    assert len(grads) == length
    elemwise_products = [
      math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
      for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
        ]

    # Second backprop  
    grads_with_none = gradients(elemwise_products, xs)
    return_grads = [
      grad_elem if grad_elem is not None \
      else tf.zeros_like(x) \
      for x, grad_elem in zip(xs, grads_with_none)]
    return return_grads

def avg_l2_dist(orig, adv):
    """Get the mean l2 distortion between two orig and adv images"""
    l2_dist = 0.0
    for i in range(orig.shape[0]):
        l2_dist+= np.linalg.norm(orig[i] - adv[i])
    return l2_dist/orig.shape[0]


def visualize(image_list, num_images):
    """Visualize images in a grid"""
    assert(len(image_list) == num_images)
    fig=plt.figure(figsize=(15,15))
    columns = num_images
    for i in range(1, columns+1):
        img = image_list[i-1]
        
        fig.add_subplot(1, columns, i)
        if img.shape[-1] == 1:
            img = np.squeeze(img)
            plt.imshow(img,cmap='Greys')
        else:
            plt.imshow(img)
        plt.axis('off')    
        
    plt.show()

#Normalize rows of a given matrix
def normalize(matrix):
    """Normalize each row vector in a matrix"""
    matrix_nm = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        norm = np.linalg.norm(matrix[i]) 
        if norm > 0:
            matrix_nm[i] = matrix[i]/np.linalg.norm(matrix[i]) 
    return matrix_nm

def preds_to_labels(preds):
    labels = np.zeros(preds.shape)
    labels[np.arange(preds.shape[0]),np.argmax(preds, axis=1)] = 1
    return labels

def norms_and_cos(model, data, labels, grads_train):
    grads = model.get_gradients_wrt_params(data, labels)
    grads_nm = normalize(grads)
    norms = np.sqrt(np.dot(grads, grads.T)).diagonal()
    cos_sim = np.dot(grads_nm, grads_train.T)
    del grads_nm, grads
    gc.collect()
    return norms, cos_sim

def greater_cos(cos_sim, eta):
    count = 0.0
    num_ = cos_sim.shape[0]
    for i in range(num_):
        if np.max(cos_sim[i]) > eta:
            count+=1.0
    return (count/num_)

def smaller_norm(norms, gamma):
    count=0.0
    num_ = norms.shape[0]
    for i in range(num_):
        if norms[i] < gamma:
            count+=1.0
    return (count/num_)

def cos_and_norm_sep(cos_sim, norms, eta, gamma):
    count=0.0
    num_ = norms.shape[0]
    for i in range(num_):
        if np.max(cos_sim[i]) > eta and norms[i] < gamma:
            count+=1.0
    return (count/num_)

def comp_cos(cos_a, cos_b):
    count = 0.0
    num_ = cos_a.shape[0]
    for i in range(num_):
        if np.max(cos_a[i]) > np.max(cos_b[i]):
            count+=1.0
    return (count/num_)

def comp_norm(norm_a, norm_b):
    count = 0.0
    num_ = norm_a.shape[0]
    for i in range(num_):
        if norm_a[i] > norm_b[i]:
            count+=1.0
    return (count/num_)

def get_test_from_train_idx(a, b):
    mask = np.ones_like(a,dtype=bool)
    mask[b] = False
    return a[mask]