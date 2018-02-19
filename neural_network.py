from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import gradients
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.examples.tutorials.mnist import input_data
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import CarliniWagnerL2

import numpy as np
import os, time, math
from keras import backend as K
from keras.models import Sequential
import influence.util as util

SEED = 12


class NeuralNetwork(object):
    """General Neural Network Class for multi-class classification """
    
    def __init__(self, model_name=None, num_classes=10, batch_size=512, initial_learning_rate=8e-1, load_from_file=False, load_model_path='', load_weights_path=''):
        """
        Desc:
            Constructor
        
        Params:
            model_name(str): Name of model (will be saved as such)
            num_classes(int): number of classes to be classified by model
            batch_size(int): Batch size to be used during training
            initial_learning_rate(float): Learning rate to start training the model. 
            load_from_file(bool): load parameters of model from file
            model_chkpt_file: tf model file containing params
                  
        """
        
        #Reproduciblity of experiments
        np.random.seed(SEED)
        tf.set_random_seed(SEED)
        
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.initial_learning_rate = initial_learning_rate
        self.data = input_data.read_data_sets("MNIST_data/", one_hot=True)
        
        
        #Hardcode the data for now. We will add better database objects in the future
        self.train_data = self.data.train.images
        self.train_labels = self.data.train.labels
        self.val_data = self.data.validation.images
        self.val_labels = self.data.validation.labels
        self.test_data = self.data.test.images
        self.test_labels = self.data.test.labels
        
        self.reshape_data()
         
        # Initialize Tf and Keras
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        K.set_session(self.sess)
       
        #Dropout and L2 reg hyperparams
        self.beta = 0.01
        self.dropout_prob = 0.5
        
        
        #Setup Input Placeholders required for forward pass, implemented by child classes (eg: CNN or FCN)
        self.input_placeholder, self.labels_placeholder, self.input_shape, self.label_shape = self.create_input_placeholders() 
                
        # Setup model operation implemented by child classes
        self.model = self.create_model()
        self.logits, self.preds = self.get_logits_preds(self.input_placeholder)
        self.params = self.get_params()
        
        num_params = 0
        for j in range(len(self.params)):
            num_params = num_params + int(np.product(self.params[j].shape))
        self.num_params = num_params
        

        # Setup loss 
        self.training_loss = self.get_loss_op(self.logits, self.labels_placeholder)

        # Setup gradients 
        self.grad_loss_wrt_param = tf.gradients(self.training_loss, self.params)
      
        #Required for HVP
        self.v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in self.params]
        self.hessian_vector = util.hessian_vector_product(self.training_loss, self.params, self.v_placeholder)
        self.grad_loss_wrt_input = tf.gradients(self.training_loss, self.input_placeholder)        

        # Because tf.gradients auto accumulates, we probably don't need the add_n (or even reduce_sum)        
        self.influence_op = tf.add_n(
              [tf.reduce_sum(tf.multiply(a, array_ops.stop_gradient(b))) for a, b in zip(self.grad_loss_wrt_param, self.v_placeholder)])

        #Required for calculation of influce of perturbation of data point on test point
        self.grad_influence_wrt_input_op = tf.gradients(self.influence_op, self.input_placeholder)
        
        #Load model parameters from file or initialize 
        if load_from_file == False:
            self.compile_model()
        else:
            self.load_model(load_model_path, load_weights_path)
        return

    def get_inverse_hvp_lissa(self, v, scale=10000.0, damping=0.0, num_samples=1, recursion_depth=1000):
        """
        Desc:
            Lissa algorithm used for approximating the inverse HVP
        Param:
            v(np.array): Vector containing gradient of loss wrt a parameter for a test point
            scale(float): scaling float for the hvp
            damping(float): 
            num_samples(int): number of times to calculate HVP for controlling variance
            recursion_depth(int): Number of training samples to use for recursively estimating HVP
        """    
        inverse_hvp = None
        print_iter = recursion_depth / 10

        for i in range(num_samples):
            samples = np.random.choice(len(self.train_data), size=recursion_depth)
           
            cur_estimate = v

            for j in range(recursion_depth):
           
                batch_xs= self.train_data[samples[j],:].reshape(self.input_shape)
                batch_ys = self.train_labels[samples[j]].reshape(self.label_shape)
                feed_dict = {self.input_placeholder: batch_xs, self.labels_placeholder: batch_ys, K.learning_phase(): 0}
                feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, cur_estimate)
                
                
                hessian_vector_val = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
                cur_estimate = [a + (1-damping) * b - c/scale for (a,b,c) in zip(v, cur_estimate, hessian_vector_val)]    
                

                # Update: v + (I - Hessian_at_x) * cur_estimate
                if (j % print_iter == 0) or (j == recursion_depth - 1):
                    feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, cur_estimate)

            if inverse_hvp is None:
                inverse_hvp = [b/scale for b in cur_estimate]
            else:
                inverse_hvp = [a + b/scale for (a, b) in zip(inverse_hvp, cur_estimate)]  

        inverse_hvp = [a/num_samples for a in inverse_hvp]
       
        return inverse_hvp 
        

    def get_influence_on_test_loss(self, train_image, train_label, inverse_hvp):
        """
        Desc:
            Calculate the influence of upweighting train_image on the network loss
            of test image used in inverse_hvp.
        Param:
            train_image(np array): training image used to calculate influence on Loss wrt a test point
            train_label(np array): one hot label of train_image
            inverse_hvp(np array): inverse hvp calculated wrt the test point
        
        """
        #Fill the feed dict with the training example 
        input_feed = train_image.reshape(self.input_shape)
        labels_feed = train_label.reshape(self.label_shape)
        
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
            K.learning_phase(): 0
        } 
                
        grad_loss_wrt_train_input = self.sess.run(self.grad_loss_wrt_param, feed_dict=feed_dict)   
        inf_of_train_on_test_loss = np.dot(np.concatenate(inverse_hvp), np.concatenate(grad_loss_wrt_train_input)) 
        return inf_of_train_on_test_loss

    
    def get_loss_op(self, logits, labels):
        """
        Desc:
            Create operation used to calculate loss during network training and for influence calculations. 
        """
        out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return out
    
    def get_gradients_wrt_params(self, X, Y):
        """Get gradients of Loss(X,Y) wrt network params"""
        
        num_params = self.num_params
        inp_size = X.shape[0]
        gradient = np.zeros((inp_size, num_params), dtype=np.float32)
        
        
        #Get one gradient at a time
        for i in range(inp_size):
            grad = self.sess.run(
                    self.grad_loss_wrt_param,
                    feed_dict={
                        self.input_placeholder: X[i].reshape(self.input_shape),                    
                        self.labels_placeholder: Y[i].reshape(self.label_shape),
                        K.learning_phase(): 0
                        }
                    )
            temp = np.array([])
            for j in range(len(grad)):
                layer_size = np.prod(grad[j].shape) 
                temp = np.concatenate((temp, np.reshape(grad[j], (layer_size))), axis=0)
        
            gradient[i,:] = temp
        return gradient
    
    def get_adversarial_version(self, x, y, eps=0.3, attack='FGSM'):
        """
        Desc:
            Caclulate the adversarial version for point x using FGSM
            x: matrix of n x input_shape samples
            y: matrix of n x input_label samples
            eps: used for FGSM
            attack: FGMS or CW
        
        """
        if attack=='FGSM':
            x_adv = np.zeros_like(x)
            for i in range(x.shape[0]): 
                feed_dict = {
                    self.input_placeholder: x[i].reshape(self.input_shape) ,
                    self.labels_placeholder: y[i].reshape(self.label_shape),
                    K.learning_phase(): 0
                } 
                grad = self.sess.run(self.grad_loss_wrt_input, feed_dict=feed_dict)[0]
                x_adv[i] = x[i] + eps*np.sign(grad[0])
            
        elif attack == 'CW':
            K.set_learning_phase(0)
            # Instantiate a CW attack object
            model = KerasModelWrapper(self.model)
            cw = CarliniWagnerL2(model, sess=self.sess)
            yname = 'y'
            cw_params = {'binary_search_steps': 1,
                 yname: None,
                 'max_iterations': 100,
                 'learning_rate': 0.1,
                 'batch_size': 1,
                 'initial_const': 10}
            x_adv = cw.generate_np(x,**cw_params)
            
        return x_adv
        
    def get_random_version(self, x, y, eps=0.3):
        """
        Desc:
            Caclulate the adversarial version for point x using FGSM
            x: n x input_shape matrix of samples to perturb
            y: n x label_shape matrix of labels
            eps: eps to use for perturbation
        
        """
        x_rand = x + eps*np.sign(np.random.uniform(low=-0.01,high=0.01,size=x.shape))
        return x_rand
    
    
    def get_inverse_hvp(self, test_point, test_label):
        """
        Desc:
            Return the inverse Hessian Vector product for test_point.
        """
        
        #Fill the feed dict with the test example
        input_feed = test_point.reshape(self.input_shape) 
        labels_feed = test_label.reshape(self.label_shape)
        
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
            K.learning_phase(): 0
        }

        v = self.sess.run(self.grad_loss_wrt_param, feed_dict=feed_dict)
        
        v_= np.array([])
        for j in range(len(v)):
            layer_size = np.prod(v[j].shape) 
            v_ = np.concatenate((v_, np.reshape(v[j], (layer_size))), axis=0)
        
        
        if (np.linalg.norm(v_)) == 0.0:
            print ('Error: Loss was 0.0 at test point %d' % test_idx)
            return
        
        
        inverse_hvp = self.get_inverse_hvp_lissa(v)
        inverse_hvp_ = np.array([])
        for j in range(len(inverse_hvp)):
            layer_size = np.prod(inverse_hvp[j].shape) 
            inverse_hvp_ = np.concatenate((inverse_hvp_, np.reshape(inverse_hvp[j], (layer_size))), axis=0)
            
        return inverse_hvp_
    
    def generate_perturbed_data(self, x, y, eps=0.3, seed=SEED, perturbation='FGSM'):
        """
        Generate a perturbed data set using FGSM, CW, or random uniform noise.
        x: n x input_shape matrix
        y: n x input_labels matrix
        seed: seed to use for reproducing experiments
        perturbation: FGSM, CW, or Noise.
        
        return:
        x_perturbed: perturbed version of x
        """
        if perturbation == 'FGSM':
            x_perturbed = self.get_adversarial_version(x,y,eps,attack='FGSM')
            
        elif perturbation == 'CW':
            x_perturbed = self.get_adversarial_version(x,y,attack='CW')
        
        else:
            x_perturbed = self.get_random_version(x,y,eps)
        
        return x_perturbed
        
    
    def gen_rand_indices_all_classes(self, y=None, seed=SEED, class_=0,num_samples=10):
        """
           Generate random indices to be used for sampling points
           y: n x label_shape matrix containing labels
           
        """
        if y is not None:
            np.random.seed(seed)
            all_class_indices = list()
            for c_ in range(self.num_classes):
                class_indices = self.gen_rand_indices_class(y,class_=class_,num_samples=num_samples) 
                all_class_indices[c_*num_samples: c_*num_samples+num_samples] = class_indices[:]
            
            return all_class_indices
        else:
            print ('Please provide training labels')
            return
        
    def gen_rand_indices_class(self, y=None, class_=0, num_samples=10):
        """
        Generate indices for the given class
        """
        if y is not None:
            c_indices = np.random.choice(np.where(np.argmax(y,axis=1) == class_)[0], num_samples)
            return c_indices
        else:
            print ('Please provide training labels')
        
    def gen_rand_indices(self, low=0, high=1000,seed=SEED, num_samples=1000):
        """
        Randomly sample indices from a range
        """
        np.random.seed(seed)
        indices = np.random.choice(range(low,high), num_samples)
        return indices
        
        
    def train(self, epochs):
        """
        Desc:
            Trains model for a specified number of epochs.
        """    
        
        self.model.fit(
            self.train_data, 
            self.train_labels,
            epochs=epochs,
            batch_size=128,
            validation_data=(self.val_data, self.val_labels),
            verbose=1,
            shuffle=True
        )

        self.model.evaluate(self.test_data, self.test_labels, batch_size=self.batch_size)
    
    def update_feed_dict_with_v_placeholder(self, feed_dict, vec):
        """
        Desc:
            Utility function for updating v_placeholder and feed_dict
        """
        for pl_block, vec_block in zip(self.v_placeholder, vec):
            feed_dict[pl_block] = vec_block  
        return feed_dict

    def evaluate_influence_matrix(self, use_grad_matrix=False, use_hvp_matrix=False, num_train_samples=1000, inf_save_path='', hvp_save_path='', grad_save_path='',grad_matrix_path='', hvp_matrix_path='',test_points_path='', test_labels_path='',seed=SEED,verbose=True):
        """
            Desc:
                Calculate influence matrix of training points on test points.
            Params:
              use_grad_matrix: load up a calculated gradient matrix or create a new matrix
              use_hvp_matrix: load up a calculated hvp matrix or create a new matrix
              num_train_samples: number of training samples to use for calculating influence
              inf_save_path: path to save the influence matrix
              hvp_save_path: path to save the hvp matrix
              grad_save_path: path to save the gradient matrix
              hvp_matrix_path: path to load up a saved hvp matrix
              grad_matrix_path: path to load up gradients matrix for training points
              seed: seed to reproduce experiments

              
          @return:
              influence_matrix: matrix of size test_points x training_points
              hvp_matrix: test_points x num_params matrix
              grad_matrix: train_points x num_params matrix
              
        """
        
        if verbose:
            print ('Getting gradients...')
            
        #Get training gradients
        if use_grad_matrix:
            train_gradients = np.load(grad_matrix_path)
        
        else:
            random_data_indices = self.gen_rand_indices(low=0,high=self.train_data.shape[0],seed=seed, num_samples=num_train_samples)
            random_train_data = self.train_data[random_data_indices]
            random_train_labels = self.train_labels[random_data_indices]
            train_gradients = self.get_gradients_wrt_params(random_train_data, random_train_labels)
        
        if verbose:
            print ('Computed gradients. Saving to file')
        if grad_save_path != '':
            np.save(grad_save_path, train_gradients)    # .npy extension is added if not given
          
        if verbose:
            print ('Saved gradients.\nGetting Inverse HVP for test points')
            
        #Perform influence calculation by loading hvp matrix or calculating it
        if use_hvp_matrix:
            hvp_matrix = np.load(hvp_matrix_path)
        
        else:
            #We need test points for which to calculate influence
            assert(os.path.isfile(test_points_path))
            assert(os.path.isfile(test_labels_path))
            test_points = np.load(test_points_path)
            test_labels = np.load(test_labels_path)
            
            hvp_matrix = np.zeros((test_points.shape[0]*self.num_classes, self.num_params))
               
            #Build up hvp_matrix
            for idx, img_lab in enumerate(zip(test_points, test_labels)):
                inverse_hvp = self.get_inverse_hvp(img_lab[0], img_lab[1])
                hvp_matrix[idx,:] = inverse_hvp
        
        if verbose:      
            print ('Done Inverse HVP.\nComputing influence\n')   
            
        #Perform dot product for influence
        inf_matrix = np.dot(hvp_matrix, train_gradients.T)
        
        if verbose:
            print ('Done computing Influence.\nSaving matrices')

        #Save results
        if inf_save_path != '':
            np.save(inf_save_path, inf_matrix)
        if hvp_save_path != '':
            np.save(hvp_save_path, hvp_matrix)
        if verbose:
            print ('Saved matrices')
            
        return inf_matrix, hvp_matrix, train_gradients
 
    def evaluate_similarity_matrix(self, use_train_grad_matrix=False, use_test_grad_matrix=False, num_train_samples=1000, sim_save_path='', test_grad_save_path='', train_grad_matrix_path='', test_grad_matrix_path='', test_points_path='', test_labels_path='',seed=SEED,verbose=True):
        """
            Desc:
                Calculate influence matrix of training points on test points.
            Params:
              use_train_grad_matrix: load up a calculated gradient matrix for training pts or create a new matrix
              use_test_grad_matrix: load up a calculated gradient matrix for test points or create a new matrix
              num_train_samples: number of training samples to use for calculating influence
              sim_save_path: path to save the influence matrix
              test_grad_save_path: path to save the gradient matrix for test points
              train_grad_save_path: path to save the gradient matrix for train points
              test_grad_matrix_path: path to load up a saved gradients matrix for test points
              train_grad_matrix_path: path to load up gradients matrix for training points
              
          @return:
              similarity_matrix: matrix of size test_points x training_points
              test_grad_matrix: test_points x num_params matrix
              train_grad_matrix: train_points x num_params matrix
              
        """
        
        if verbose:
            print ('Getting gradients for training points...')
                   
        #Get training gradients
        if use_train_grad_matrix:
            train_gradients = np.load(train_grad_matrix_path)
        
        else:
            random_data_indices = self.gen_rand_indices(low=0,high=self.train_data.shape[0],seed=seed, num_samples=num_train_samples)
            random_train_data = self.train_data[random_data_indices]
            random_train_labels = self.train_labels[random_data_indices]
            train_gradients = self.get_gradients_wrt_params(random_train_data, random_train_labels)
        
        if verbose:
            print ('Computed gradients. Saving to file')
                   
        if train_grad_save_path != '':
            np.save(train_grad_save_path, train_gradients)
        if verbose:
            print ('Saved gradients.\nGetting gradients for test points')
        
        if use_test_grad_matrix:
            test_gradients = np.load(test_grad_matrix_path)
   
        else:
            #We need test points for which to calculate influence
            assert(os.path.isfile(test_points_path))
            assert(os.path.isfile(test_labels_path))
            test_points = np.load(test_points_path)
            test_labels = np.load(test_labels_path)
            
            test_gradients = np.array(self.get_gradients_wrt_params(test_points, test_labels))
        if verbose:   
            print ('Done computing test gradients.\nComputing similarity')   
            
        #Perform dot product
        sim_matrix = np.dot(test_gradients, train_gradients.T)
        
        if verbose:       
            print ('Done similarity.\nSaving matrices')

        #Save results
        if sim_save_path != '':
            np.save(sim_save_path, sim_matrix)
        if test_grad_save_path != '':
            np.save(test_grad_save_path,test_gradients)
        if verbose:
            print ('Done saving matrices.')
            
        return sim_matrix, test_gradients, train_gradients
    
  