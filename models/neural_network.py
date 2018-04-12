from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import gradients
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops
from keras.datasets import cifar10, mnist
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import CarliniWagnerL2, BasicIterativeMethod, DeepFool, SaliencyMapMethod

import numpy as np
import os, time, math
from keras import backend as K
from keras.models import Sequential
from keras.utils import np_utils

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
            X_train, Y_train, X_test, Y_test, x_extra, y_extra = np.load('../data/svhn_data.npy')
            
            X_train = np.transpose(X_train, (3,0,1,2))
            X_test = np.transpose(X_test, (3,0,1,2))
            x_extra = np.transpose(x_extra, (3,0,1,2))

            Y_train = Y_train.reshape((Y_train.shape[0], ))
            Y_test = Y_test.reshape((Y_test.shape[0], ))
            y_extra = y_extra.reshape((y_extra.shape[0],))
            
            X_train = np.concatenate((X_train, x_extra), axis=0)
            Y_train = np.concatenate((Y_train, y_extra), axis=0)

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
    
    
    def create_wb_fgsm_graph(self):
        """Create a TF graph that will be used for Whitebox FGSM and return grad operation"""
        #Create/modify TF graph
        x_guide = tf.placeholder(dtype= tf.float32, shape=self.input_shape)
        y_guide = tf.placeholder(dtype = tf.float32, shape=self.label_shape)
            
        guide_logits, guide_preds = self.get_logits_preds(x_guide)
        guide_loss = self.get_loss_op(guide_logits, y_guide)
            
        guide_grad = tf.gradients(guide_loss, self.params)
        x_grad = self.grad_loss_wrt_param
            
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
        loss_total = self.training_loss + loss_sim + loss_norm
        total_grad = tf.gradients(loss_total, self.input_placeholder)   
        
        return x_guide, y_guide, total_grad
        
    def fgsm_attack(self, x, y=None, eps=0.3, clip_min=0.0, clip_max=1.0):
        """Create FGSM attack points for each row of x"""
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
        return x_adv
    
    def visualize(self, image):
        import matplotlib
        import matplotlib.pyplot as plt
        
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


    
    def fgsm_wb_attack(self, x, y=None, eps=0.3, clip_min=0.0, clip_max=1.0, x_tar=None, y_tar=None, x_guide=None, y_guide=None, total_grad=None):
        """Create Whitebox FGSM attack points """
        x_adv = np.zeros_like(x)
            
        #Iterate over all samples to be perturbed
        for i in range(x.shape[0]): 
            feed_dict = {
                self.input_placeholder: x[i].reshape(self.input_shape) ,
                self.labels_placeholder: y[i].reshape(self.label_shape),
                x_guide: x_tar[i].reshape(self.input_shape), 
                y_guide: y_tar[i].reshape(self.label_shape),
                K.learning_phase(): 0
             } 
            grad = self.sess.run(total_grad, feed_dict=feed_dict)[0]
            x_adv_raw = x[i] + eps*np.sign(grad[0])
            x_adv[i] = x_adv_raw.clip(clip_min, clip_max)
            
        return x_adv
    
    def get_adversarial_version(self, x, y=None, eps=0.3, iterations=10000,attack='FGSM', targeted=False, x_tar=None, y_tar=None,clip_min=0.0, clip_max = 1.0, use_cos_norm_reg=False, nb_candidate=10):
        """
        Desc:
            Caclulate the adversarial version for point x using FGSM
            x: matrix of n x input_shape samples
            y: matrix of n x input_label samples
            eps: used for FGSM
            attack: FGMS or CW
        
        """
        if attack=='FGSM':
            x_adv = self.fgsm_attack(x,y, eps=eps, clip_min=clip_min, clip_max=clip_max)
       
        if attack=='FGSM-WB':
            #Create matrix to store adversarial samples
            x_guide, y_guide, total_grad = self.create_wb_fgsm_graph()
            x_adv = self.fgsm_wb_attack(x,y=y,eps=eps,clip_min=clip_min, clip_max=clip_max,x_tar=x_tar, y_tar=y_tar, x_guide=x_guide, y_guide=y_guide, total_grad=total_grad)
        
        #We implement BIM-A as cleverhans does not directly support it
        elif attack=='BIM-A':
            x_adv = np.zeros_like(x)
            eps_ = eps / float(iterations)
            for i in range(x.shape[0]): 
                x_adv_clipped = x[i]
                for k in range(iterations):
                    x_curr = np.array([x_adv_clipped])
                    y_curr = np.array([y[i]])
                    x_adv_clipped = self.fgsm_attack(x_curr,y=y_curr, eps=eps_,clip_min=clip_min, clip_max=clip_max)[0]
                    #Check if the label has changed. Stop if so.
                    if np.argmax(self.model.predict(x_adv_clipped.reshape(*self.input_shape))) != np.argmax(y[i]):
                        break
                x_adv[i] = x_adv_clipped
                        
        elif attack == 'BIM-B':
            x_adv = np.zeros_like(x)
            eps_ = eps / float(iterations)
            for i in range(x.shape[0]): 
                x_adv_clipped = x[i]
                for k in range(iterations):
                    x_curr = np.array([x_adv_clipped])
                    y_curr = np.array([y[i]])
                    x_adv_clipped = self.fgsm_attack(x_curr,y=y_curr, eps=eps_,clip_min=clip_min, clip_max=clip_max)[0]
                x_adv[i] = x_adv_clipped
                
        elif attack=='BIM-A-WB':
            x_adv = np.zeros_like(x)
            x_guide, y_guide, total_grad = self.create_wb_fgsm_graph()
            eps_ = eps / float(iterations)
            for i in range(x.shape[0]): 
                x_adv_clipped = x[i]
                for k in range(iterations):
                    x_curr = np.array([x_adv_clipped])
                    y_curr = np.array([y[i]])
                    x_tar_curr = np.array([x_tar[i]])
                    y_tar_curr = np.array([y_tar[i]])
                    x_adv_clipped = self.fgsm_wb_attack(x_curr,y=y_curr, eps=eps_,clip_min=clip_min, clip_max=clip_max,x_tar=x_tar_curr, y_tar=y_tar_curr,x_guide=x_guide, y_guide=y_guide, total_grad=total_grad)[0]
                    #Check if the label has changed. Stop if so.
                    if np.argmax(self.model.predict(x_adv_clipped.reshape(*self.input_shape))) != np.argmax(y[i]):
                        break
                x_adv[i] = x_adv_clipped

        
                        
        elif attack=='BIM-B-WB':
            x_adv = np.zeros_like(x)
            x_guide, y_guide, total_grad = self.create_wb_fgsm_graph()
            eps_ = eps / float(iterations)
            for i in range(x.shape[0]): 
                x_adv_clipped = x[i]
                for k in range(iterations):
                    x_curr = np.array([x_adv_clipped])
                    y_curr = np.array([y[i]])
                    x_tar_curr = np.array([x_tar[i]])
                    y_tar_curr = np.array([y_tar[i]])
                    x_adv_clipped = self.fgsm_wb_attack(x_curr,y=y_curr, eps=eps_,clip_min=clip_min, clip_max=clip_max,x_tar=x_tar_curr, y_tar=y_tar_curr,x_guide=x_guide, y_guide=y_guide, total_grad=total_grad)[0]
                x_adv[i] = x_adv_clipped   
        
        elif attack == 'CW':
            K.set_learning_phase(0)
            # Instantiate a CW attack object
            model = KerasModelWrapper(self.model)
            cw = CarliniWagnerL2(model, sess=self.sess)
            adv_inputs = x
            
            if targeted:
                adv_ys = y_tar
                guide_inp = x_tar
                yname = "y_target"
                use_cos_norm_reg = use_cos_norm_reg
            else:     
                yname = 'y'
                adv_ys = None
                guide_inp = None
                use_cos_norm_reg = use_cos_norm_reg
                   
            cw_params = {'binary_search_steps': 1,
                 'abort_early': False,
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
    
    
    def generate_perturbed_data(self, x, y=None, eps=0.3, iterations=100,seed=SEED, perturbation='FGSM', targeted=False, x_tar=None,y_tar=None, use_cos_norm_reg=False, nb_candidate=10):
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
            x_perturbed = self.get_adversarial_version(x,attack='CW',iterations=iterations,eps=eps, targeted=targeted, x_tar=x_tar, y_tar=y_tar,use_cos_norm_reg=use_cos_norm_reg)
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
    
