from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import gradients
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops
from keras.datasets import cifar10, mnist
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import CarliniWagnerL2, BasicIterativeMethod

import numpy as np
import os, time, math
from keras import backend as K
from keras.models import Sequential
from keras.utils import np_utils
import influence.util as util

SEED = 12


class NeuralNetwork(object):
    """General Neural Network Class for multi-class classification """
    
    def __init__(self, model_name=None, dataset='mnist', batch_size=512, initial_learning_rate=8e-1, load_from_file=False, load_model_path='', load_weights_path=''):
        """
        Desc:
            Constructor
        
        Params:
            model_name(str): Name of model (will be saved as such)
            dataset(str): Name of dataset to load - also determines which model will be loaded
            batch_size(int): Batch size to be used during training
            initial_learning_rate(float): Learning rate to start training the model. 
            load_from_file(bool): load parameters of model from file
            model_chkpt_file: tf model file containing params
                  
        """
        
        #Reproduciblity of experiments
        np.random.seed(SEED)
        tf.set_random_seed(SEED)
        
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.initial_learning_rate = initial_learning_rate
        
        if 'mnist' in dataset.lower():
            self.num_classes = 10
            self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.load_dataset('mnist')
        
        elif 'svhn' in dataset.lower():
            self.num_classes = 10
            self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.load_dataset('svhn')
                      
        elif 'cifar2' in dataset.lower():
            self.num_classes = 2
            self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.load_dataset('cifar2')
        
        
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
        self.model = self.create_model(dataset.lower())
        self.logits, self.preds = self.get_logits_preds(self.input_placeholder)
        self.params = self.get_params()
        
        #Get total number of params in model
        num_params = 0
        for j in range(len(self.params)):
            num_params = num_params + int(np.product(self.params[j].shape))
        self.num_params = num_params

        # Setup loss 
        self.training_loss = self.get_loss_op(self.logits, self.labels_placeholder)

        # Setup gradients 
        self.grad_loss_wrt_param = tf.gradients(self.training_loss, self.params)
        self.grad_loss_wrt_input = tf.gradients(self.training_loss, self.input_placeholder)        
        
        #Load model parameters from file or initialize 
        if load_from_file == False:
            self.compile_model()
        else:
            self.load_model(load_model_path, load_weights_path)
        return

    
    def load_dataset(self, dataset='mnist'):
        """
        Desc: Load the required dataset into the model
        """
        
        if dataset == 'mnist':
            (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
            X_train = X_train.reshape(-1, 28, 28, 1)
            X_test = X_test.reshape(-1, 28, 28, 1)
            
            Y_train = np_utils.to_categorical(Y_train, 10)
            Y_test = np_utils.to_categorical(Y_test, 10)
            
            self.input_side = 28
            self.input_channels = 1
            self.input_dim = self.input_side * self.input_side * self.input_channels
            
            
            
        elif dataset == 'svhn':
            X_train, Y_train, X_test, Y_test, _, _ = np.load('./svhn_data.npy')

            X_train = np.transpose(X_train, (3,0,1,2))
            X_test = np.transpose(X_test, (3,0,1,2))

            Y_train = Y_train.reshape((Y_train.shape[0], ))
            Y_test = Y_test.reshape((Y_test.shape[0], ))

            Y_train[np.where(Y_train == 10)] = 0
            Y_test[np.where(Y_test == 10)] = 0

            Y_train_cat = np.zeros((Y_train.shape[0], 10))
            Y_test_cat = np.zeros((Y_test.shape[0], 10))

            Y_train_cat[range(Y_train.shape[0]), Y_train] = 1
            Y_test_cat[range(Y_test.shape[0]), Y_test] = 1

            Y_train = Y_train_cat
            Y_test = Y_test_cat
            
            self.input_side = 32
            self.input_channels = 3
            self.input_dim = self.input_side * self.input_side * self.input_channels
            
        elif dataset == 'cifar2':
            #Convert cifar10 into cifar2
            (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
            X_train = X_train.reshape(-1, 32, 32, 3)
            X_test = X_test.reshape(-1, 32, 32, 3)
            Y_train = Y_train.reshape((Y_train.shape[0],))
            Y_test = Y_test.reshape((Y_test.shape[0],))
                       
            #Pick two classes at random and update dataset
            c1, c2 = np.random.choice(range(10), 2)[:]

            X_train = X_train[np.where((Y_train == c1) | (Y_train == c2))[0]]
            Y_train = Y_train[np.where((Y_train == c1) | (Y_train == c2))[0]]

            X_test = X_test[np.where((Y_test == c1) | (Y_test == c2))[0]]
            Y_test = Y_test[np.where((Y_test == c1) | (Y_test == c2))[0]]

            #Update labels
            Y_train[np.where(Y_train == c1)] = 0
            Y_train[np.where(Y_train == c2)] = 1

            Y_test[np.where(Y_test == c1)] = 0
            Y_test[np.where(Y_test == c2)] = 1
            
            #Convert to one hot
            Y_train = np_utils.to_categorical(Y_train, 2)
            Y_test = np_utils.to_categorical(Y_test, 2)
            
            self.input_side = 32
            self.input_channels = 3
            self.input_dim = self.input_side * self.input_side * self.input_channels
            
    
        #Normalize data
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        
        num_val = int(X_test.shape[0]/2.0)
        
        #Get validation sets as well
        val_indices = np.random.choice(range(X_test.shape[0]), num_val)
        X_val = X_test[val_indices]
        Y_val = Y_test[val_indices]
    
        mask = np.ones(X_test.shape[0], dtype=bool)
        mask[val_indices] = False
    
        X_test = X_test[mask]
        Y_test = Y_test[mask]
            
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
        

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
    
    def get_gradients_wrt_input(self, X, Y):
        """Get gradients of Loss(X,Y) wrt input"""
        
        inp_size = X.shape[0]
        inp_shape = np.prod(X[0].shape)
        gradient = np.zeros((inp_size, inp_shape), dtype=np.float32)
        
        #Get one gradient at a time
        for i in range(inp_size):
            grad = self.sess.run(
                    self.grad_loss_wrt_input,
                    feed_dict={
                        self.input_placeholder: X[i].reshape(self.input_shape),                    
                        self.labels_placeholder: Y[i].reshape(self.label_shape),
                        K.learning_phase(): 0
                        }
                    )[0]
            gradient[i,:] = grad.reshape(inp_shape)
        return gradient
    
    def get_adversarial_version(self, x, y, eps=0.3, iterations=100000,attack='FGSM', targeted=False, x_tar=None, clip_min=0.0, clip_max = 1.0, use_cos_norm_reg=False):
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
                x_adv_raw = x[i] + eps*np.sign(grad[0])
                x_adv[i] = x_adv_raw.clip(clip_min, clip_max)
            
        elif attack == 'CW':
            K.set_learning_phase(0)
            # Instantiate a CW attack object
            model = KerasModelWrapper(self.model)
            cw = CarliniWagnerL2(model, sess=self.sess)
            adv_inputs = x
            
            if targeted:
                adv_ys = y
                guide_inp = x_tar
                yname = "y_target"
                use_cos_norm_reg = use_cos_norm_reg
            else:     
                yname = 'y'
                adv_ys = None
                guide_inp = None
                use_cos_norm_reg = use_cos_norm_reg
                   
            cw_params = {'binary_search_steps': 1,
                 yname: adv_ys,
                 'guide_img': guide_inp,
                 'max_iterations': iterations,
                 'use_cos_norm_reg': use_cos_norm_reg,
                 'learning_rate': 0.1,
                 'batch_size': 1,
                  'clip_min': clip_min,
                  'clip_max': clip_max,
                 'initial_const': 10}     
                
              
            x_adv = cw.generate_np(adv_inputs,**cw_params)
        
        elif attack == 'BIM':
            K.set_learning_phase(0)
            model = KerasModelWrapper(self.model)
            bim = BasicIterativeMethod(model, sess=self.sess)
            yname = 'y'
            bim_params = {'eps': eps,
                          yname: None,
                          'eps_iter': eps/float(iterations),
                          'nb_iter': iterations,
                          'clip_min': clip_min,
                          'clip_max': clip_max}
            x_adv = bim.generate_np(x, **bim_params)
        return x_adv
        
    def get_random_version(self, x, y, eps=0.3,min_clip=0.0, max_clip=1.0):
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
    
    
    def generate_perturbed_data(self, x, y, eps=0.3, iterations=100,seed=SEED, perturbation='FGSM', targeted=False, x_tar=None, use_cos_norm_reg=False):
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
            x_perturbed = self.get_adversarial_version(x,y,attack='FGSM', eps=eps)
            
        elif perturbation == 'CW':
            x_perturbed = self.get_adversarial_version(x,y,attack='CW',iterations=iterations,eps=eps, targeted=targeted, x_tar=x_tar, use_cos_norm_reg=use_cos_norm_reg)
        elif perturbation == 'BIM':
            x_perturbed = self.get_adversarial_version(x,y,attack='BIM',iterations=iterations,eps=eps)
        else:
            x_perturbed = self.get_random_version(x,y,eps)
        
        return x_perturbed
        
    
    def gen_rand_indices_all_classes(self, y=None, seed=SEED,num_samples=10):
        """
           Generate random indices to be used for sampling points
           y: n x label_shape matrix containing labels
           
        """
        if y is not None:
            np.random.seed(seed)
            all_class_indices = list()
            for c_ in range(self.num_classes):
                class_indices = self.gen_rand_indices_class(y,class_=c_,num_samples=num_samples) 
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
    
