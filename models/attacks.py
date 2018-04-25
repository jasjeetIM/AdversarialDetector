import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.utils import np_utils

#Random Attacks
def get_random_version(model, x, y, eps=0.3,min_clip=0.0, max_clip=1.0):
    """
    Desc:
      Caclulate the adversarial version for point x using FGSM
      x: n x input_shape matrix of samples to perturb
      y: n x label_shape matrix of labels
      eps: eps to use for perturbation
    
    """
    x_rand_raw = x + eps*np.sign(np.random.uniform(low=-0.01,high=0.01,size=x.shape))
    x_rand = x_rand_raw.clip(min_clip, max_clip)
    return x_rand
  
def get_random_version_binary(model, x, y, eps = 10, min_clip=0.0, max_clip=1.0):
    """
    Desc:
      Caclulate the adversarial version for point x using FGSM
      x: n x input_shape matrix of samples to perturb
      y: n x label_shape matrix of labels
      eps: eps to use for perturbation
    
    """
    x_rand = np.zeros_like(x)
    for i in range(x.shape[0]):
        x_rand_raw = x[i]
        rand_idx = np.random.choice(range(model.input_shape[1]), eps)
        x_rand_raw[rand_idx]+=1
        x_rand[i] = x_rand_raw % 2    
    return x_rand

#Binary Attacks


def jsma_symbolic_update(model, x_in, y_in,domain_in, clip_min, clip_max):
    preds = model.model(x_in)
    preds_onehot = tf.one_hot(tf.argmax(preds, axis=1), depth=model.num_classes)
  
    #We always increase features
    increase = True
    
    # create the Jacobian graph
    list_derivatives = []
    for class_ind in xrange(model.num_classes):
        derivatives = tf.gradients(preds[:, class_ind], x_in)
        list_derivatives.append(derivatives[0])
    grads = tf.reshape(tf.stack(list_derivatives),
               shape=[model.num_classes, -1, model.input_shape[1]])
    # Compute the Jacobian components
    # To help with the computation later, reshape the target_class
    # and other_class to [nb_classes, -1, 1].
    # The last dimention is added to allow broadcasting later.
    target_class = tf.reshape(tf.transpose(y_in, perm=[1, 0]),
                  shape=[model.num_classes, -1, 1])
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

    other_tmp = grads_other
    other_tmp += increase_coef \
      * tf.reduce_max(tf.abs(grads_other), axis=1, keep_dims=True)

    # Create a mask to only keep features that match conditions
    if increase:
        scores_mask = ((target_tmp > 0) & (other_tmp < 0))
    else:
        scores_mask = ((target_tmp < 0) & (other_tmp > 0))

   
    scores = tf.cast(scores_mask, tf.float32) * (-target_tmp * other_tmp)
    # Extract the best pixel
    best = tf.argmax(
          tf.reshape(scores, shape=[-1, model.input_shape[1]]),axis=1)[0]
   
    p1_one_hot = tf.one_hot(best, depth=model.input_shape[1])
 
    # Update the search domain
    domain_out = domain_in - p1_one_hot

    # Apply the modification to the images
    if increase:
        #with tf.control_dependencies([p1_one_hot]):
        x_out = tf.minimum(clip_max, x_in + p1_one_hot)
    else:
        x_out = tf.maximum(clip_min, x_in - p1_one_hot)
  
    return x_out, domain_out, preds
  
def jsma_binary(model, x, y=None, clip_min=0.0, clip_max=1.0, iterations=100):
    search_domain = tf.placeholder(dtype=tf.float32, shape=model.input_shape)
    labels_placeholder = tf.cast(model.labels_placeholder,tf.float32)
    x_out, domain_out, preds = jsma_symbolic_update(model, model.input_placeholder, labels_placeholder, search_domain, clip_min, clip_max)
 
    X_adv = np.zeros_like(x)
    avg_iterations = 0
    for i in range(x.shape[0]):
        x_i = x[i]
        y_ = y[i]
        d_i = (x_i != 1).astype(int)  
        for j in range(iterations): 
            feed_dict={
              model.input_placeholder: x_i.reshape(model.input_shape),
              model.labels_placeholder: y_.reshape(model.label_shape), 
              search_domain: d_i.reshape(model.input_shape),
              K.learning_phase():0 
            }
      
            #Get updated values for input and search domain
            x_i, d_i, pred = model.sess.run([x_out, domain_out, preds], feed_dict=feed_dict)
            if (np.argmax(y_) == np.argmax(pred)):
                break
            #Update return matrix
        X_adv[i] = x_i
    return X_adv 
  
def fgsm_attack_binary(model, x, y=None, clip_min=0.0, clip_max=1.0):
    """Create FGSM attack points for each row of x"""
    X_adv = np.zeros_like(x)
    gradients = model.grad_loss_wrt_input
    adv = tf.sign(gradients)
    #Get all non negative indices
    neg = tf.constant(-1, shape=(model.input_shape),dtype=tf.float32)
    pos_idx = tf.cast(tf.not_equal(adv, neg), dtype=tf.float32)
    #Add 1 to all indices that will make the loss go up
    x_adv_raw = model.input_placeholder + pos_idx
    #Clip feature vectors to be 0 or 1
    x_adv = tf.mod(x_adv_raw, 2)
    
    for i in range(x.shape[0]): 
        feed_dict = {
          model.input_placeholder: x[i].reshape(model.input_shape) ,
          model.labels_placeholder: y[i].reshape(model.label_shape),
          K.learning_phase(): 0
         } 
        x_adversarial = model.sess.run(x_adv, feed_dict=feed_dict)[0]
        X_adv[i] = x_adversarial
    return X_adv
  
def bim_a_attack_binary(model, x, y=None, iterations=10,clip_min=0.0, clip_max=1.0):
    """Create BIM attack points for each row of x"""
    X_adv = np.zeros_like(x)
    gradients = model.grad_loss_wrt_input
    adv = tf.sign(gradients)
    #Get feature with highest derivative
    best = tf.argmax(tf.reshape(gradients, shape=(model.input_shape)), axis=1)[0]
    p1 = tf.one_hot(best, depth=model.input_shape[1])
    x_adv_raw = model.input_placeholder + p1
    #Clip feature vectors to be 0 or 1
    x_adv = tf.mod(x_adv_raw, 2)

    for i in range(x.shape[0]):
        x_i = x[i]
        y_i = y[i]    
        for j in range(iterations):
            feed_dict={
              model.input_placeholder: x_i.reshape(model.input_shape),
              model.labels_placeholder: y_i.reshape(model.label_shape),
              K.learning_phase():0
            }
            x_i = model.sess.run(x_adv ,feed_dict=feed_dict)[0]
            if np.argmax(model.model.predict(x_i.reshape(*model.input_shape))) != np.argmax(y[i]):
                break  
        X_adv[i] = x_i
    
    return X_adv
  
def bim_b_attack_binary(model, x, y=None, iterations=10,clip_min=0.0, clip_max=1.0):
    """Create BIM attack points for each row of x"""
    X_adv = np.zeros_like(x)
    gradients = model.grad_loss_wrt_input
    adv = tf.sign(gradients)
    #Get feature with highest derivative
    best = tf.argmax(tf.reshape(gradients, shape=(model.input_shape)), axis=1)[0]
    p1 = tf.one_hot(best, depth=model.input_shape[1])
    x_adv_raw = model.input_placeholder + p1
    #Clip feature vectors to be 0 or 1
    x_adv = tf.mod(x_adv_raw, 2)

    for i in range(x.shape[0]):
        x_i = x[i]
        y_i = y[i]    
        for j in range(iterations):
            feed_dict={
              model.input_placeholder: x_i.reshape(model.input_shape),
              model.labels_placeholder: y_i.reshape(model.label_shape),
              K.learning_phase():0
            }
            x_i = model.sess.run(x_adv ,feed_dict=feed_dict)[0]
        
        X_adv[i] = x_i
    
    return X_adv

#Regular Attacks
def fgsm_attack(model, x, y=None, eps=0.3, clip_min=0.0, clip_max=1.0):
    """Create FGSM attack points for each row of x"""
    x_adv = np.zeros_like(x)
    for i in range(x.shape[0]): 
        feed_dict = {
          model.input_placeholder: x[i].reshape(model.input_shape) ,
          model.labels_placeholder: y[i].reshape(model.label_shape),
          K.learning_phase(): 0
         } 
        grad = model.sess.run(model.grad_loss_wrt_input, feed_dict=feed_dict)[0]
        x_adv_raw = x[i] + eps*np.sign(grad[0])
        x_adv[i] = x_adv_raw.clip(clip_min, clip_max)
    return x_adv
  
def bim_a_attack(model, x, y=None, eps=0.3, clip_min=0.0, clip_max=1.0, iterations=10):
    x_adv = np.zeros_like(x)
    eps_ = eps / float(iterations)
    for i in range(x.shape[0]): 
        x_adv_clipped = x[i]
        for k in range(iterations):
            x_curr = np.array([x_adv_clipped])
            y_curr = np.array([y[i]])
            x_adv_clipped = fgsm_attack(model, x_curr,y=y_curr, eps=eps_,clip_min=clip_min, clip_max=clip_max)[0]
            #Check if the label has changed. Stop if so.
            if np.argmax(model.model.predict(x_adv_clipped.reshape(*model.input_shape))) != np.argmax(y[i]):
                break
        x_adv[i] = x_adv_clipped
    return x_adv
  
def bim_b_attack(model, x, y=None, eps=0.3, clip_min=0.0, clip_max=1.0, iterations=10):
    x_adv = np.zeros_like(x)
    eps_ = eps / float(iterations)
    for i in range(x.shape[0]): 
        x_adv_clipped = x[i]
        for k in range(iterations):
            x_curr = np.array([x_adv_clipped])
            y_curr = np.array([y[i]])
            x_adv_clipped = fgsm_attack(model,x_curr,y=y_curr, eps=eps_,clip_min=clip_min, clip_max=clip_max)[0]
        x_adv[i] = x_adv_clipped
    return x_adv

def create_wb_fgsm_graph(model):
    """Create a TF graph that will be used for Whitebox FGSM and return grad operation"""
    #Create/modify TF graph
    x_guide = tf.placeholder(dtype= tf.float32, shape=model.input_shape)
    y_guide = tf.placeholder(dtype = tf.float32, shape=model.label_shape)
            
    guide_logits, guide_preds = model.get_logits_preds(x_guide)
    guide_loss = model.get_loss_op(guide_logits, y_guide)
            
    guide_grad = tf.gradients(guide_loss, model.params)
    x_grad = model.grad_loss_wrt_param
            
    for j in range(len(guide_grad)):
        if j == 0:
            temp_x = tf.reshape(x_grad[j], [-1])
            temp_guide = tf.reshape(guide_grad[j], [-1])
        else:
            temp_x_t = tf.reshape(x_grad[j], [-1])
            temp_guide_t = tf.reshape(guide_grad[j], [-1])
            temp_x = tf.concat([temp_x, temp_x_t], 0)
            temp_guide = tf.concat([temp_guide, temp_guide_t], 0)

    x_grad = temp_x
    guide_grad = temp_guide
    #Define cos sim based loss
    loss_sim = tf.reduce_sum(
                            tf.multiply(x_grad, guide_grad)) / (tf.norm(x_grad)*tf.norm(guide_grad))
    loss_norm = -tf.norm(x_grad)
    loss_total = model.training_loss + loss_sim + loss_norm
    total_grad = tf.gradients(loss_total, model.input_placeholder)   
        
    return x_guide, y_guide, total_grad


def fgsm_wb_attack(model, x, y=None, eps=0.3, clip_min=0.0, clip_max=1.0, x_tar=None, y_tar=None, x_guide=None, y_guide=None, total_grad=None):
    """Create Whitebox FGSM attack points """
    x_adv = np.zeros_like(x)
      
    #Iterate over all samples to be perturbed
    for i in range(x.shape[0]): 
        feed_dict = {
            model.input_placeholder: x[i].reshape(model.input_shape) ,
            model.labels_placeholder: y[i].reshape(model.label_shape),
            x_guide: x_tar[i].reshape(model.input_shape), 
            y_guide: y_tar[i].reshape(model.label_shape),
            K.learning_phase(): 0
         } 
        grad = model.sess.run(total_grad, feed_dict=feed_dict)[0]
        x_adv_raw = x[i] + eps*np.sign(grad[0])
        x_adv[i] = x_adv_raw.clip(clip_min, clip_max)
      
    return x_adv
  
def bim_a_wb_attack(model, x, y=None, eps=0.3, clip_min=0.0, clip_max=1.0, x_tar=None, y_tar=None, iterations=10):
    x_adv = np.zeros_like(x)
    x_guide, y_guide, total_grad = create_wb_fgsm_graph(model)
    eps_ = eps / float(iterations)
    for i in range(x.shape[0]): 
        x_adv_clipped = x[i]
        for k in range(iterations):
            x_curr = np.array([x_adv_clipped])
            y_curr = np.array([y[i]])
            x_tar_curr = np.array([x_tar[i]])
            y_tar_curr = np.array([y_tar[i]])
            x_adv_clipped = fgsm_wb_attack(model,x_curr,y=y_curr, eps=eps_,clip_min=clip_min, clip_max=clip_max,x_tar=x_tar_curr, y_tar=y_tar_curr,x_guide=x_guide, y_guide=y_guide, total_grad=total_grad)[0]
            #Check if the label has changed. Stop if so.
            if np.argmax(model.model.predict(x_adv_clipped.reshape(*model.input_shape))) != np.argmax(y[i]):
                break
        x_adv[i] = x_adv_clipped
    return x_adv
  
def bim_b_wb_attack(model, x, y=None, eps=0.3, clip_min=0.0, clip_max=1.0, x_tar=None, y_tar=None, iterations=10):
    x_adv = np.zeros_like(x)
    x_guide, y_guide, total_grad = create_wb_fgsm_graph(model)
    eps_ = eps / float(iterations)
    for i in range(x.shape[0]): 
        x_adv_clipped = x[i]
        for k in range(iterations):
            x_curr = np.array([x_adv_clipped])
            y_curr = np.array([y[i]])
            x_tar_curr = np.array([x_tar[i]])
            y_tar_curr = np.array([y_tar[i]])
            x_adv_clipped = fgsm_wb_attack(model,x_curr,y=y_curr, eps=eps_,clip_min=clip_min, clip_max=clip_max,x_tar=x_tar_curr, y_tar=y_tar_curr,x_guide=x_guide, y_guide=y_guide, total_grad=total_grad)[0]
        x_adv[i] = x_adv_clipped
    return x_adv