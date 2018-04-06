from tensorflow import gradients
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops
import sys
import numpy as np

## Visualization of samples
#import matplotlib
#import matplotlib.pyplot as plt

def hessian_vector_product(ys, xs, v):
  """ Multiply the Hessian of `ys` wrt `xs` by `v`.
  This is an efficient construction that uses a backprop-like approach
  to compute the product between the Hessian and another vector. The
  Hessian is usually too large to be explicitly computed or even
  represented, but this method allows us to at least multiply by it
  for the same big-O cost as backprop.
  Implicit Hessian-vector products are the main practical, scalable way
  of using second derivatives with neural networks. They allow us to
  do things like construct Krylov subspaces and approximate conjugate
  gradient descent.
  Example: if `y` = 1/2 `x`^T A `x`, then `hessian_vector_product(y,
  x, v)` will return an expression that evaluates to the same values
  as (A + A.T) `v`.
  Args:
    ys: A scalar value, or a tensor or list of tensors to be summed to
        yield a scalar.
    xs: A list of tensors that we should construct the Hessian over.
    v: A list of tensors, with the same shapes as xs, that we want to
       multiply by the Hessian.
  Returns:
    A list of tensors (or if the list would be length 1, a single tensor)
    containing the product between the Hessian and `v`.
  Raises:
    ValueError: `xs` and `v` have different length.
  """ 

  # Validate the input
  length = len(xs)
  if len(v) != length:
    raise ValueError("xs and v must have the same length.")

  # First backprop
  grads = gradients(ys, xs)

  # grads = xs

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

#Only works on 100 x n matrices right now
def analyze_matrix(data_matrix, preds, train_labels='',num_classes=10,str_type='Sim', verbose=False):
    """
    Desc: Compares two influence or simiarity matrices
    data_matrix: sim/influence scores in a 100 x n matrix where each row represents a test point and n = # train points
    pred: predictions of the network on 100 test points used in data_matrix. Each prediction is a scalar.
    train_labels: labels of all training points used in calculation of sim/inf. 
    """
    if train_labels == '':
        return
    
    #Count of times max,mean,var of max inf/sim class matched predicted class
    max_match_pred_class = 0.0
    mean_match_pred_class = 0.0
    var_match_pred_class = 0.0
    min_match_pred_class = 0.0
    
    
    #Iterate over each class
    for c in range(num_classes):
        
    #Iterate over each point in each class
      for pts in range(10):
        test_idx = c*10 + pts
        if verbose:
            print ('\nTest Class : %d, Test Pred: %d' % (c, preds[test_idx]) )
            
        #We need the below to get the max values and max classes for this test point
        max_val = -sys.float_info.max
        min_val = sys.float_info.max
        max_mean_val = -sys.float_info.max
        max_var_val = -sys.float_info.max
        
        max_val_class = -1
        max_mean_val_class = -1
        max_var_val_class = -1
        min_val_class = -1
       
        #Iterate over all classes
        for tc in range(num_classes):
            #Get classes indices for training points in curr class
            class_indices = np.where(train_labels == tc)
            scores = data_matrix[test_idx, class_indices]
            smax = np.max(scores)
            smin = np.min(scores)
            mean = np.mean(scores)
            median = np.median(scores)
            var = np.var(scores)
            if verbose: 
                prt_st = '    Train Class: %d, Max ' + str_type + ': %.8f, Min ' + str_type + ': %.8f, Mean ' + str_type + ': %.8f, Median ' + str_type + ': %.8f, Var ' + str_type + ': %.8f'
                print (prt_st % (tc, smax, smin, mean, median, var ))
                
            #Check if this class has higher values than previous classes
            if smax > max_val:
                max_val = smax
                max_val_class = tc
            if smin < min_val:
                min_val = smin
                min_val_class = tc
            if mean > max_mean_val:
                max_mean_val = mean
                max_mean_val_class = tc
            if var > max_var_val:
                max_var_val = var
                max_var_val_class = tc
        
        #Update counts of max values matching predicted classes 
        if max_val_class == preds[test_idx]:
            max_match_pred_class+=1
        if max_mean_val_class == preds[test_idx]:
            mean_match_pred_class+=1
        if max_var_val_class == preds[test_idx]:
            var_match_pred_class+=1
        if min_val_class == preds[test_idx]:
            min_match_pred_class+=1
        if verbose: 
            prt_st = '\nSummary: Max '+ str_type + ': %.8f, %d. \tMin ' + str_type + ': %.8f, %d. \tMax Mean: %.8f, %d. \tMax Var: %.8f,%d'
            print (prt_st % (max_val, max_val_class, min_val, min_val_class, max_mean_val, max_mean_val_class, max_var_val, max_var_val_class))
                
    print ('\nMax Match: %.3f, Min Match : %.3f, Mean Match: %.3f, Var Match : %.3f' % (max_match_pred_class/100.0,min_match_pred_class/100.0, mean_match_pred_class/100.0, var_match_pred_class/100.0)  )            

    
#Only works on 100 x n matrices right now
def compare_matrices(matrix_1, matrix_2, pred_1, pred_2,train_labels='',num_classes=10,str_type='Sim', verbose=False):
    """
    matrix_1: sim/influence scores in a 100 x n matrix where each row represents a test point and n = # train points
    matrix_2: sim/influence scores in a 100 x n matrix where each row represents a test point and n = # train points
    pred_1: predictions of the network on 100 test points used in matrix_1. Each prediction is a scalar.
    pred_2: predictions of the network on 100 test points used in matrix_2 
    
    Note: Ratios are based on matrix_1_values/matrix_2_values 
    
    """
    
    #Get total training points
    total_pts = matrix_1.shape[1]
    (total_pts)
    
    #Counts of the number of times matrix_1 values were higher than matrix_2 values for all classes
    max_higher_all_classes = 0.0
    mean_higher_all_classes = 0.0
    var_higher_all_classes = 0.0
    median_higher_all_classes = 0.0
    min_lower_all_classes = 0.0
    
    #Counts of the number of times matrix_1 values were higher than matrix_2 for the MAX class
    max_higher_max_class = 0.0
    mean_higher_max_class = 0.0
    var_higher_max_class = 0.0
    median_higher_max_class = 0.0
    min_lower_max_class = 0.0
    
    
    #Counts of the number of times matrix_1 values were higher than matrix_2 values for all classes
    max_all_classes_ratio = list()
    mean_all_classes_ratio = list()
    var_all_classes_ratio = list()
    median_all_classes_ratio = list()
    min_all_classes_ratio = list()
    
    #Counts of the number of times matrix_1 values were higher than matrix_2 for the MAX class
    max_max_class_ratio = list()
    mean_max_class_ratio = list()
    var_max_class_ratio = list()
    median_max_class_ratio = list()
    min_max_class_ratio = list()
    
    for c in range(num_classes):
        
    #Iterate over each point in each class
      for pts in range(10):
        test_idx = c*10 + pts
        if verbose:
            print ('\nTest Class : %d, Test Pred 1: %d, Test Pred 2: %d' % (c, pred_1[test_idx], pred_2[test_idx]) )
        
        #Iterate over each class in the training points
        max_val1 = -sys.float_info.max
        max_mean_val1 = -sys.float_info.max
        max_var1 = -sys.float_info.max
        max_median1 = -sys.float_info.max
        min_val1 = sys.float_info.max
        
        max_val_class1 = -1
        max_mean_val_class1 = -1
        max_var_class1 = -1
        max_median_class1 = -1
        min_val_class1 = -1
        
        max_val2 = -sys.float_info.max
        max_mean_val2 = -sys.float_info.max
        max_var2 = -sys.float_info.max
        max_median2 = -sys.float_info.max
        min_val2=sys.float_info.max

        
        max_val_class2 = -1
        max_mean_val_class2 = -1
        max_var_class2 = -1
        max_median_class2 = -1
        min_val_class2 = -1
       
        for tc in range(num_classes):
            #Get training points for this class
            class_indices = np.where(train_labels == tc)
            val_scores1 = matrix_1[test_idx, class_indices]
            val_scores2 = matrix_2[test_idx, class_indices]
            smax1 = np.max(val_scores1)
            smin1 = np.min(val_scores1)
            mean1 = np.mean(val_scores1)
            median1 = np.median(val_scores1)
            var1 = np.var(val_scores1)
            smax2 = np.max(val_scores2)
            smin2 = np.min(val_scores2)
            mean2 = np.mean(val_scores2)
            median2 = np.median(val_scores2)
            var2 = np.var(val_scores2)

            
            max_all_classes_ratio.append(smax1/smax2)
            mean_all_classes_ratio.append(mean1/mean2)
            var_all_classes_ratio.append(var1/var2)
            median_all_classes_ratio.append(median1/median2)
            min_all_classes_ratio.append(smin1/smin2)

            
            #Compare matrix_1 values to matrix_2 values
            if smax1 > smax2:
                max_higher_all_classes+=1.0
            if mean1 > mean2:
                mean_higher_all_classes+=1.0
            if median1 > median2:
                median_higher_all_classes +=1.0
            if var1 > var2:
                var_higher_all_classes+=1.0
            if smin1 < smin2:
                min_lower_all_classes+=1.0
                
                
            if verbose:
                prt_st = '    Train Class: %d, Max ' + str_type + ' Ratio: %.8f, Min ' + str_type + ' Ratio: %.8f, Mean ' + str_type + ' Ratio: %.8f, Median ' + str_type + ' Ratio: %.8f, Var ' + str_type + ' Ratio: %.8f'
                print (prt_st % (tc, smax1/smax2, smin1/smin2, mean1/mean2, median1/median2, var1/var2 ))
            
            #Update max values for each matrix
            if smax1 > max_val1:
                max_val1 = smax1
                max_val_class1 = tc
            if mean1 > max_mean_val1:
                max_mean_val1 = mean1
                max_mean_val_class1 = tc
            if var1 > max_var1:
                max_var1 = var1
                max_var_class1 = tc
            if median1 > max_median1:
                max_median1 = median1
                max_median_class1 = tc
            if smin1 < min_val1:
                min_val1 = smin1
                min_val_class1 = tc
            
            if smax2 > max_val2:
                max_val2 = smax2
                max_val_class2 = tc
            if mean2 > max_mean_val2:
                max_mean_val2 = mean2
                max_mean_val_class2 = tc
            if var2 > max_var2:
                max_var2 = var2
                max_var_class2 = tc
            if median2 > max_median2:
                max_median2 = median2
                max_median_class2 = tc
            if smin2 < min_val2:
                min_val2 = smin2
                min_val_class2 = tc
            
        max_max_class_ratio.append(max_val1/max_val2)
        mean_max_class_ratio.append(max_mean_val1/max_mean_val2)
        var_max_class_ratio.append(max_var1/max_var2)
        median_max_class_ratio.append(max_median2/max_median2)
        min_max_class_ratio.append(min_val1/min_val2)     
        
        #Compare matrix_1 max values to matrix_2 max values
        if max_val1 > max_val2:
            max_higher_max_class+=1.0
        if max_mean_val1 > max_mean_val2:
            mean_higher_max_class+=1.0
        if max_var1 > max_var2:
            var_higher_max_class+=1.0
        if max_median1 > max_median2:
            median_higher_max_class+=1.0
        if min_val1 < min_val2:
            min_lower_max_class+=1.0
        
        if verbose:
            prt_st = '\nSummary: Max '+ str_type + ' Ratio: %.8f, %d, %d.\tMin '+ str_type + ' Ratio: %.8f, %d, %d.\tMax Mean Ratio: %.8f, %d, %d. \tMax Var Ratio: %.8f,%d,%d'
            print (prt_st % (max_val1/max_val2, max_val_class1, max_val_class2, min_val1/min_val2, min_val_class1, min_val_class2, max_mean_val1/max_mean_val2, max_mean_val_class1, 
                             max_mean_val_class2, max_var1/max_var2, max_var_class1, max_var_class2))

    print ('Max Higher All: %.3f, Min Lower All: %.3f Mean Higher All : %.3f, Median Higher All: %.2f, Var Higher All; %.2f' % (max_higher_all_classes/float(1000), min_lower_all_classes/float(1000),
                                                                                                                              
                                                                                                            mean_higher_all_classes/float(1000),
                                                                                                            median_higher_all_classes/float(1000),
                                                                                                            var_higher_all_classes/float(1000)))
    print ('Max All Ratio: Mean=%.3f, Var=%.3f, Mean All Ratio: Mean=%.3f, Var=%.3f, Median All Ratio: Mean=%.3f, Var=%.3f, Var All Ratio: Mean=%.3f, Var=%.3f' % (np.mean(max_all_classes_ratio), np.var(max_all_classes_ratio), np.mean(mean_all_classes_ratio), np.var(mean_all_classes_ratio),np.mean(median_all_classes_ratio), np.var(median_all_classes_ratio),np.mean(var_all_classes_ratio), np.var(var_all_classes_ratio)))
    
    print ('Max Higher Max: %.3f, Min Lower Max: %.3f, Mean Higher Max : %.3f, Median Higher Max: %.2f, Var Higher Max; %.2f' % (max_higher_max_class/float(100), min_lower_max_class/float(100),
                                                                                                            mean_higher_max_class/float(100),
                                                                                                            median_higher_max_class/float(100),
                                                                                                            var_higher_max_class/float(100)))
    print ('Max Max Ratio: Mean=%.3f, Var=%.3f, Mean Max Ratio: Mean=%.3f, Var=%.3f, Median Max Ratio: Mean=%.3f, Var=%.3f, Var Max Ratio: Mean=%.3f, Var=%.3f' % (np.mean(max_max_class_ratio), np.var(max_max_class_ratio), np.mean(mean_max_class_ratio), np.var(mean_max_class_ratio),np.mean(median_max_class_ratio), np.var(median_max_class_ratio),np.mean(var_max_class_ratio), np.var(var_max_class_ratio)))
        
    
"""
def visualize(image):
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
    """