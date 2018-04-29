from __future__ import division
from __future__ import print_function

import sys, os
sys.path.append('../')
from attacks import *
from util import hessian_vector_product

import tensorflow as tf
from keras.datasets import cifar10, mnist
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import CarliniWagnerL2, BasicIterativeMethod, DeepFool, SaliencyMapMethod

import numpy as np
import os, time, math, gc
from keras import backend as K
from keras.models import Sequential
from keras.utils import np_utils

class NeuralNetwork(object):
    """General Neural Network Class for multi-class classification """
    SEED = 14
    
    def __init__(self, model_name=None, dataset='mnist', batch_size=512, initial_learning_rate=8e-1, load_from_file=False, load_model_path='', load_weights_path='', seed=14):
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
        SEED = seed
        #Reproduciblity of experiments
        np.random.seed(SEED)
        tf.set_random_seed(SEED)
        
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.initial_learning_rate = initial_learning_rate
        
        if 'mnist' in dataset.lower():
            self.num_classes = 10
            self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.load_dataset('mnist')
        
        elif 'drebin' in dataset.lower():
            self.num_classes = 2
            self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.load_dataset('drebin')
                      
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
        
        self.v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in self.params]
        self.hessian_vector = hessian_vector_product(self.training_loss, self.params, self.v_placeholder)
        
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
            
            
            
        elif dataset == 'drebin':
            X_train, Y_train, X_val, Y_val, X_test, Y_test = self.load_drebin_data()
            
            self.input_side = 1
            self.input_channels = 1
            #Hardcode dimensionality
            self.input_dim = 545333
            
        elif dataset == 'cifar2':
            #Convert cifar10 into cifar2
            (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
            X_train = X_train.reshape(-1, 32, 32, 3)
            X_test = X_test.reshape(-1, 32, 32, 3)
            Y_train = Y_train.reshape((Y_train.shape[0],))
            Y_test = Y_test.reshape((Y_test.shape[0],))
                       
            #Pick two classes at random and update dataset
            c1, c2 = np.random.choice(range(10), 2)[:]
            print ('Using classes: %d, and %d' % (c1, c2))

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
    
    def load_drebin_data(self, ben_path= '../data/ben_matrix.npy', mal_path='../data/mal_matrix.npy'):
        ben_data = np.load(ben_path)
        mal_data = np.load(mal_path)
    
        ben_lab = np.zeros((ben_data.shape[0]), dtype=np.int8)
        mal_lab = np.ones((mal_data.shape[0]), dtype=np.int8)
        ben_lab = np_utils.to_categorical(ben_lab, 2)
        mal_lab = np_utils.to_categorical(mal_lab, 2)
    
    
        X = np.concatenate((ben_data, mal_data), axis=0)
        Y = np.concatenate((ben_lab, mal_lab), axis=0)
        del ben_data, mal_data, ben_lab, mal_lab
        gc.collect()
    
        num_samples = X.shape[0]
    
        
        num_val = int(.05*num_samples)
        num_test = int(.05*num_samples)
        num_train = num_samples - num_val - num_test
    
        reshuffle_idx = np.random.choice(range(X.shape[0]), num_samples,replace=False)
        X = X[reshuffle_idx]
        Y = Y[reshuffle_idx]
    
        X_train = X[:num_train]
        Y_train = Y[:num_train]
    
        X_val = X[num_train:num_train+num_val]
        Y_val = Y[num_train:num_train+num_val]
    
        X_test = X[num_train+num_val:]
        Y_test = Y[num_train+num_val:]
    
        return X_train, Y_train, X_val, Y_val, X_test, Y_test        

    def get_loss_op(self, logits, labels):
        """
        Desc:
            Create operation used to calculate loss during network training and for influence calculations. 
        """
        out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return out
    
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
    
    def get_inverse_hvp_lissa(self, v, scale=10000.0, damping=0.0, num_samples=1, recursion_depth=100, seed=SEED):
        """
        Desc:
            Lissa algorithm used for approximating the inverse HVP
        Param:
            v(np.array): Vector containing gradient of loss wrt a parameters at a test point
            scale(float): scaling float for the hvp
            damping(float): 
            num_samples(int): number of times to calculate HVP for controlling variance
            recursion_depth(int): Number of training samples to use for recursively estimating HVP
        """    
        inverse_hvp = None
        print_iter = recursion_depth / 10

        for i in range(num_samples):
            np.random.seed(seed)
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
    
    def update_feed_dict_with_v_placeholder(self, feed_dict, vec):
        """
        Desc:
            Utility function for updating v_placeholder and feed_dict
        """
        for pl_block, vec_block in zip(self.v_placeholder, vec):
            feed_dict[pl_block] = vec_block  
        return feed_dict
    
    
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
    
    
    def get_adversarial_version_binary(self, x, y=None, eps=10, iterations=10, attack='JSMA',clip_min=0.0, clip_max = 1.0):
        """
        Desc:
            Caclulate the adversarial version for point x using FGSM
            x: matrix of n x input_shape samples
            y: matrix of n x input_label samples
            eps: used for FGSM
            attack: FGMS or CW
        
        """
        if attack=='FGSM':
            x_adv = fgsm_attack_binary(self, x,y, clip_min=clip_min, clip_max=clip_max)
       
        elif attack=='BIM-A':
            x_adv = bim_a_attack_binary(self,x,y, iterations=iterations, clip_min=clip_min, clip_max=clip_max)
                        
        elif attack == 'BIM-B':
            x_adv = bim_b_attack_binary(self,x,y, iterations=iterations, clip_min=clip_min, clip_max=clip_max)
                
        elif attack == 'JSMA':
             x_adv = jsma_binary(self,x,y, iterations=iterations, clip_min=clip_min, clip_max=clip_max)
            
        return x_adv    
        
    def get_adversarial_version(self, x, y=None, eps=0.3, iterations=10000,attack='FGSM', targeted=False, x_tar=None, y_tar=None,clip_min=0.0, clip_max = 1.0, use_cos_norm_reg=False, use_logreg=False, num_logreg=0,nb_candidate=10, train_grads=None, num_params=100):
        """
        Desc:
            Caclulate the adversarial version for point x using FGSM
            x: matrix of n x input_shape samples
            y: matrix of n x input_label samples
            eps: used for FGSM
            attack: FGMS or CW
        
        """
        if attack=='FGSM':
            x_adv = fgsm_attack(self,x,y, eps=eps, clip_min=clip_min, clip_max=clip_max)
       
        if attack=='FGSM-WB':
            #Create matrix to store adversarial samples
            x_guide, y_guide, total_grad = create_wb_fgsm_graph(self)
            x_adv = fgsm_wb_attack(self,x,y=y,eps=eps,clip_min=clip_min, clip_max=clip_max,x_tar=x_tar, y_tar=y_tar, x_guide=x_guide, y_guide=y_guide, total_grad=total_grad)
        
        #We implement BIM-A as cleverhans does not directly support it
        elif attack=='BIM-A':
            x_adv = bim_a_attack(self,x,y, eps=eps, clip_min=clip_min, clip_max=clip_max)
                        
        elif attack == 'BIM-B':
            x_adv = bim_b_attack(self,x,y, eps=eps, clip_min=clip_min, clip_max=clip_max)
                
        elif attack=='BIM-A-WB':
            x_adv = bim_a_wb_attack(self,x,y=y,eps=eps,clip_min=clip_min, clip_max=clip_max,x_tar=x_tar, y_tar=y_tar)
                        
        elif attack=='BIM-B-WB':
            x_adv = bim_a_wb_attack(self,x,y=y,eps=eps,clip_min=clip_min, clip_max=clip_max,x_tar=x_tar, y_tar=y_tar)
        
        elif attack == 'CW':
            K.set_learning_phase(0)
            # Instantiate a CW attack object
            model = KerasModelWrapper(self.model)
            cw = CarliniWagnerL2(model, sess=self.sess)
            adv_inputs = x
            if targeted:
                adv_ys = y_tar
                if use_cos_norm_reg:
                    guide_inp = x_tar
                else:
                    guide_inp = None
                yname = "y_target"
            else:     
                yname = 'y'
                adv_ys = None
                guide_inp = None
                
                
            cw_params = {'binary_search_steps': 1,
                 'abort_early': True,
                  yname: adv_ys,
                 'guide_img': guide_inp,
                 'max_iterations': iterations,
                 'use_cos_norm_reg': use_cos_norm_reg,
                 'use_logreg': use_logreg,
                 'train_grads': train_grads,
                 'num_logreg': num_logreg,
                 'num_params': num_params,
                 'learning_rate': 0.1,
                 'batch_size': 1,
                  'clip_min': clip_min,
                  'clip_max': clip_max,
                 'initial_const': 10,}     

            x_adv = cw.generate_np(adv_inputs,**cw_params)
            
        elif attack == 'DF':
            K.set_learning_phase(0)
            model = KerasModelWrapper(self.model)
            df = DeepFool(model, sess=self.sess)
            df_params = {'nb_candidate': nb_candidate}
            x_adv = df.generate_np(x,**df_params)

        elif attack == 'JSMA':
            K.set_learning_phase(0)
            model = KerasModelWrapper(self.model)
            jsma = SaliencyMapMethod(model,sess=self.sess)
            jsma_params = {'theta': 1., 
                           'gamma': 0.1,
                           'clip_min': clip_min, 
                           'clip_max': clip_max,
                           'y_target': None}
            x_adv = jsma.generate_np(x, **jsma_params)
            
        return x_adv
        
    
    def generate_perturbed_data(self, x, y=None, eps=0.3, iterations=50,seed=SEED, perturbation='FGSM', targeted=False, x_tar=None,y_tar=None, use_cos_norm_reg=False,use_logreg=False, num_logreg=0, nb_candidate=10, train_grads=None, num_params=100):
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
        elif perturbation == 'FGSM-WB':
            x_perturbed = self.get_adversarial_version(x,y,attack='FGSM-WB', eps=eps, x_tar=x_tar, y_tar=y_tar)     
        elif perturbation == 'CW':
            x_perturbed = self.get_adversarial_version(x,attack='CW',iterations=iterations,eps=eps, targeted=targeted, x_tar=x_tar, y_tar=y_tar,use_cos_norm_reg=use_cos_norm_reg,use_logreg=use_logreg, num_logreg=num_logreg, train_grads=train_grads, num_params=num_params)
        elif perturbation == 'BIM-A':
            x_perturbed = self.get_adversarial_version(x,y,attack='BIM-A',iterations=iterations,eps=eps)
        elif perturbation == 'BIM-A-WB':
            x_perturbed = self.get_adversarial_version(x,y,attack='BIM-A-WB',iterations=iterations,eps=eps, x_tar=x_tar, y_tar=y_tar)
        elif perturbation == 'BIM-B':
            x_perturbed = self.get_adversarial_version(x,y,attack='BIM-B',iterations=iterations,eps=eps)
        elif perturbation == 'BIM-B-WB':
            x_perturbed = self.get_adversarial_version(x,y,attack='BIM-B-WB',iterations=iterations,eps=eps, x_tar=x_tar, y_tar=y_tar)
        elif perturbation == 'DF':
            x_perturbed = self.get_adversarial_version(x,y,attack='DF', nb_candidate=nb_candidate)
        elif perturbation == 'JSMA':
            x_perturbed = self.get_adversarial_version(x,y,attack='JSMA')
        elif perturbation == 'NOISY':
            x_perturbed = get_random_version(self, x,y,eps)
        
        return x_perturbed
    
    def generate_perturbed_data_binary(self, x, y=None, iterations=10,seed=SEED, perturbation='FGSM', targeted=False):
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
            x_perturbed = self.get_adversarial_version_binary(x,y,attack='FGSM')
        elif perturbation == 'BIM-A':
            x_perturbed = self.get_adversarial_version_binary(x,y,attack='BIM-A', iterations=iterations)     
        elif perturbation == 'BIM-B':
            x_perturbed = self.get_adversarial_version_binary(x,y,attack='BIM-B',iterations=iterations)
        elif perturbation == 'JSMA':
            x_perturbed = self.get_adversarial_version_binary(x,y,attack='JSMA',iterations=iterations)
        elif perturbation == 'NOISY':
            x_perturbed = get_random_version_binary(self,x,y, eps=iterations)
        
        return x_perturbed    
    
    
    #Random sampling from dataset
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
        