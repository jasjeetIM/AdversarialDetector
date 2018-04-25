import numpy as np
import matplotlib, gc
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