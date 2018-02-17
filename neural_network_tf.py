from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import gradients
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import os, time, math
from keras import backend as K

import influence.util as util

SEED = 12


class NeuralNetwork(object):
    """General Neural Network Class for multi-class classification """
    
    def __init__(self, model_name=None, num_classes=10, batch_size=50, initial_learning_rate=8e-1, load_from_file=False, model_chkpt_file=''):
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
        self.train_data = self.data.train
        self.val_data = self.data.validation
        self.test_data = self.data.test
         
        # Initialize Tf and Keras
        config = tf.ConfigProto()        
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)
       
        #We do not use learning_rate decay as we rely on the Adam optimizer to do so
        #Store number of training steps applied to the network
        self.global_step = tf.Variable(0, name='global_step', trainable=False) 
        self.beta = tf.Variable(0.01, name='lambda', trainable=False)
        self.dropout_prob = tf.placeholder_with_default(0.0, shape=())
        
        
        #Setup Input Placeholders required for forward pass, implemented by child classes (eg: CNN or FCN)
        self.input_placeholder, self.labels_placeholder = self.create_input_placeholders() 
        
                
        # Setup forward pass operation, implemented by child classes
        self.logits = self.forward_pass(self.input_placeholder)
        self.preds = self.predict(self.logits)
        self.params = self.get_params()


        # Setup loss 
        self.training_loss, self.cross_entropy = self.get_loss_op(self.logits, self.labels_placeholder, self.params, self.beta)
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(tf.argmax(self.logits, 1), tf.argmax(self.labels_placeholder, 1))
        
        
        #Training the network
        self.train_op = self.create_train_op(self.training_loss, self.global_step, self.initial_learning_rate)
        
    
        # Saver to save and load models
        self.saver = tf.train.Saver()

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
            init = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            self.sess.run(init)
            self.sess.run(init_l)
        else:
            self.saver.restore(self.sess, model_chkpt_file)
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
            samples = np.random.choice(len(self.train_data.images), size=recursion_depth)
           
            cur_estimate = v

            for j in range(recursion_depth):
           
                batch_xs= self.train_data.images[samples[j],:].reshape(1,-1)
                batch_ys = self.train_data.labels[samples[j]].reshape(-1)
                feed_dict = {self.input_placeholder: batch_xs, self.labels_placeholder: batch_ys}
                feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, cur_estimate)
                
                
                hessian_vector_val = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
                cur_estimate = [a + (1-damping) * b - c/scale for (a,b,c) in zip(v, cur_estimate, hessian_vector_val)]    
                

                # Update: v + (I - Hessian_at_x) * cur_estimate
                if (j % print_iter == 0) or (j == recursion_depth - 1):
                    #print("Recursion at depth %s: norm is %.8lf" % (j, np.linalg.norm(np.concatenate(cur_estimate))))
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
        input_feed = train_image.reshape(1, -1)
        labels_feed = train_label.reshape(-1)
        
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        } 
                
        grad_loss_wrt_train_input = self.sess.run(self.grad_loss_wrt_param, feed_dict=feed_dict)   
        inf_of_train_on_test_loss = np.dot(np.concatenate(inverse_hvp), np.concatenate(grad_loss_wrt_train_input)) 
        return inf_of_train_on_test_loss
    
    def create_train_op(self, training_loss, global_step, learning_rate):
        """
        Desc: 
            Create training operation 
        """
        optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.99)
        train_op = optimizer.minimize(training_loss, global_step=global_step)
        return train_op
    
    def create_precision_op(self, logits, labels):
        """
        Desc: 
            Create operation used to calculate accuracy during network training
        """
        correct = tf.nn.in_top_k(logits, tf.argmax(labels,1), 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32)) / tf.shape(labels)[0]
    
    def get_loss_op(self, logits, labels, weights, l):
        """
        Desc:
            Create operation used to calculate loss during network training and for influence calculations. 
        """
        cross_entropy = -tf.reduce_sum(tf.multiply(tf.cast(labels, tf.float32), tf.nn.log_softmax(logits)), reduction_indices=1)
        net_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean') 
        reg_loss = 0.0
        for i in weights:
            reg_loss += l * tf.nn.l2_loss(i)
        loss = net_loss + reg_loss
        return loss, cross_entropy
    
    def get_adversarial_version(self, x, y, eps=0.3):
        """
        Desc:
            Caclulate the adversarial version for point x using FGSM
        
        """
        
        feed_dict = {
            self.input_placeholder: x.reshape(1,-1),
            self.labels_placeholder: y.reshape(-1),
        } 
        grad = self.sess.run(self.grad_loss_wrt_input, feed_dict=feed_dict)[0]
        x_adv = x + eps*np.sign(grad)
        return x_adv
    
    def get_inverse_hvp(self, test_point, test_label):
        """
        Desc:
            Return the inverse Hessian Vector product for test_point.
        """
        
        #Fill the feed dict with the test example
        input_feed = test_point.reshape(1,-1) 
        labels_feed = test_label.reshape(-1)
        
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }

        v = self.sess.run(self.grad_loss_wrt_param, feed_dict=feed_dict)
        
        if (np.linalg.norm(np.concatenate(v))) == 0.0:
            print ('Error: Loss was 0.0 at test point %d' % test_idx)
            return
        
        
        inverse_hvp = self.get_inverse_hvp_lissa(v)
        return inverse_hvp
    
    
    
    def train(self, epochs):
        """
        Desc:
            Trains model for a specified number of epochs.
        """    
        
        sess = self.sess    
        num_steps = int(np.shape(self.train_data.images)[0]/self.batch_size)

        for e in range(epochs):
            for step in xrange(num_steps):
                x,y = self.train_data.next_batch(self.batch_size)
                feed_dict = {self.dropout_prob: 0.5, self.input_placeholder: x,
                         self.labels_placeholder: y}
                _, loss = sess.run([self.train_op, self.training_loss], feed_dict=feed_dict)  
            
                if step % 100 == 0:
                 # Print status to stdout.
                    #grad = sess.run(self.grad_loss_wrt_param, feed_dict={self.input_placeholder: x[0].reshape(1,-1),
                    #                                                self.labels_placeholder:y[0].reshape(-1)})
                    #print (np.linalg.norm(np.concatenate(grad)))
                    feed_dict= {self.input_placeholder: self.val_data.images,
                            self.labels_placeholder: self.val_data.labels}
                    acc, acc_op = sess.run([self.accuracy, self.accuracy_op], feed_dict=feed_dict)
                    print (acc_op)
                    print('Acc: %.3f , Loss = %.3f ' % (acc, loss))

                # Save a checkpoint and evaluate the model periodically.
                #if (step + 1) % 100000 == 0 or (step + 1) == num_steps:
                #    if save_checkpoints: self.saver.save(sess, self.checkpoint_file, global_step=step)
                #    if verbose: self.print_model_eval()

    
    def update_feed_dict_with_v_placeholder(self, feed_dict, vec):
        """
        Desc:
            Utility function for updating v_placeholder and feed_dict
        """
        for pl_block, vec_block in zip(self.v_placeholder, vec):
            feed_dict[pl_block] = vec_block  
        return feed_dict

    def evaluate_influence_matrix(self, test_classes=[], train_classes=[], num_test_samples=10, use_adv_sample=False, use_adv_label=False, save_path='inf_matrix.dat', verbose=True):
        """
            Desc:
                Calculate influence matrix of training points on test points.
            Params:
              test_classes(list): List containing test classses for which to calculate influence for
              train_classes(list): List containing training classes used for calculating infuence on each element of test_classes
              num_test_samples(int): number of test samples to use per class
              use_adv_sample(bool): Convert samples to adversarial using FGSM if true
              use_adv_label(bool): Treat network prediction on adversarial sample as the True label while calculating influence
              
          @return:
              influence_matrix: matrix of size test_classes x train_classes 
        """

        if test_classes == []:
            #Matrix of form num_classes x num_classes, where ij represent influence of training points from
            #class j on test points from class i
            if train_classes == []:
                influence_matrix = np.zeros((self.num_classes, self.num_classes)) 
                train_classes = range(0,self.num_classes)
            else:
                influence_matrix = np.zeros((self.num_classes, len(train_classes)))
        
            test_classes = range(0,self.num_classes)
        else:
            if train_classes == []:
                influence_matrix = np.zeros((len(test_classes), self.num_classes))
                train_classes = range(0,self.num_classes)
            else:
                influence_matrix = np.zeros((len(test_classes), len(train_classes)))
        
       
        #Get number of samples per class
        samples_per_class = dict()

        #Stores number of samples per class in order to average over influence
        for c_ in train_classes:
            total_c_samples = np.sum(np.argmax(self.train_data.labels, axis=1) == c_)
            samples_per_class[c_] = float(total_c_samples)

        #Perform calculation for every class
        for cdx, c in enumerate(test_classes):
            #Get all data points from class and pick a random subset of test points
            data_points = np.where(np.argmax(self.test_data.labels, axis=1) == c)[0]
            test_class_indices = np.random.choice(data_points, num_test_samples)
        
            #Store results of influence for this test class
            inf_on_test_class = np.zeros((len(train_classes)))
        
            #Iterate over test points for the current class 
            for test_idx in test_class_indices:
                test_image = self.test_data.images[test_idx]
                test_label = self.test_data.labels[test_idx]
                
                #Convert to adversarial sample
                if use_adv_sample == True:
                    test_image = self.get_adversarial_version(test_image, test_label)
                
                feed_dict = {self.input_placeholder: test_image.reshape(1,-1), self.labels_placeholder: test_label.reshape(-1)}
                test_pred = self.sess.run(self.preds, feed_dict=feed_dict)[0]
                
                #Use adversarial prediction as a label in calculating the Inverse HVP (s_test)
                if use_adv_label == True:
                    pred_idx = np.argmax(test_pred)
                    test_label = np.zeros(test_pred.shape)
                    test_label[pred_idx] = 1.0
                
                test_pred = np.argmax(test_pred)
                test_lb = np.argmax(test_label)
            
                #Calculate the Inverse HVP for this test point
                inverse_hvp = self.get_inverse_hvp(test_image, test_label)
                
                #Evaluate influence on the test point for each training class 
                for c_dx, c_ in enumerate(train_classes):
                    train_class_indices = list(np.where(np.argmax(self.train_data.labels, axis=1) == c_)[0])
                    train_class_inf_on_pt = 0.0
                    max_inf = 0.0
                    min_inf = 0.0
                    max_idx = 0
                    #Iterate over all training points for current training class
                    for train_idx in train_class_indices:
                        train_image = self.train_data.images[train_idx]                        
                        train_label = self.train_data.labels[train_idx]
                    
                        inf = self.get_influence_on_test_loss(train_image, train_label, inverse_hvp)
                        train_class_inf_on_pt += inf
                        if inf > max_inf:
                            max_inf = inf
                            max_idx = train_idx
                        if inf < min_inf:
                            min_inf = inf
                            min_idx = train_idx
                
                    #Update influence of train class on test point
                    inf_on_test_class[c_dx] += train_class_inf_on_pt / samples_per_class[c_]
                    
                    if verbose:  
                        print ('Test Idx: %d, Test Label: %d, Test Pred: %d, Train Class: %d, Avg Inf: %.8f, Max Inf: %.8f, Max Idx: %d, Min Inf: %.8f, Min Idx: %d' % (test_idx, test_lb, test_pred, c_, inf_on_test_class[c_dx], max_inf, max_idx, min_inf, min_idx))
                
            #Divide influence by number of test samples to get average influence on test class
            inf_on_test_class /= float(num_test_samples)
            
            #Update influence matrix
            influence_matrix[cdx,:] = inf_on_test_class[:]
    

        #Save results
        influence_matrix.tofile(save_path)
        return influence_matrix
    
    def evaluate_similarity_matrix(self, test_classes=[], train_classes=[], num_test_samples=10, use_adv_sample=False, use_adv_label=False, save_path='sim_matrix.dat', verbose=True):
        """
            Desc:
                Calculate similarity between gradient of loss wrt of test_classes and gradient of loss wrt train classes
            Params:
              test_classes(list): List containing test classses for which to calculate influence for
              train_classes(list): List containing training classes used for calculating infuence on each element of test_classes
              num_test_samples(int): number of test samples to use per class
              use_adv_sample(bool): Convert samples to adversarial using FGSM if true
              use_adv_label(bool): Treat network prediction on adversarial sample as the True label while calculating influence
              
          @return:
              similarity_matrix: matrix of size test_classes x train_classes 
        """

        if test_classes == []:
            #Matrix of form num_classes x num_classes, where ij represent influence of training points from
            #class j on test points from class i
            if train_classes == []:
                similarity_matrix = np.zeros((self.num_classes, self.num_classes)) 
                train_classes = range(0,self.num_classes)
            else:
                similarity_matrix = np.zeros((self.num_classes, len(train_classes)))
        
            test_classes = range(0,self.num_classes)
        else:
            if train_classes == []:
                similarity_matrix = np.zeros((len(test_classes), self.num_classes))
                train_classes = range(0,self.num_classes)
            else:
                similarity_matrix = np.zeros((len(test_classes), len(train_classes)))
        
       
        #Get number of samples per class
        samples_per_class = dict()

        #Stores number of samples per class in order to average over influence
        for c_ in train_classes:
            total_c_samples = np.sum(np.argmax(self.train_data.labels, axis=1) == c_)
            samples_per_class[c_] = float(total_c_samples)

        #Perform calculation for every class
        for cdx, c in enumerate(test_classes):
            #Get all data points from class and pick a random subset of test points
            data_points = np.where(np.argmax(self.test_data.labels, axis=1) == c)[0]
            test_class_indices = np.random.choice(data_points, num_test_samples)
        
            #Store results of influence for this test class
            sim_with_test_class = np.zeros((len(train_classes)))
        
            #Iterate over test points for the current class 
            for test_idx in test_class_indices:
                test_image = self.test_data.images[test_idx]
                test_label = self.test_data.labels[test_idx]
                
                #Convert to adversarial sample
                if use_adv_sample == True:
                    test_image = self.get_adversarial_version(test_image, test_label)
                
                feed_dict = {self.input_placeholder: test_image.reshape(1,-1), self.labels_placeholder: test_label.reshape(-1)}
                test_pred = self.sess.run(self.preds, feed_dict=feed_dict)[0]
                
                #Use adversarial prediction as a label in calculating the Inverse HVP (s_test)
                if use_adv_label == True:
                    pred_idx = np.argmax(test_pred)
                    test_label = np.zeros(test_pred.shape)
                    test_label[pred_idx] = 1.0
            
                    feed_dict = {self.input_placeholder: test_image.reshape(1,-1), self.labels_placeholder: test_label.reshape(-1)}
                
                test_pred = np.argmax(test_pred)
                test_lb = np.argmax(test_label)
            
                #Calculate gradient of loss wrt this test sample
                grad_loss_wrt_test = np.concatenate(self.sess.run(self.grad_loss_wrt_param , feed_dict=feed_dict))

                #Evaluate the gradient vectors for all training classes in question
                for c_dx, c_ in enumerate(train_classes):
                    train_class_indices = list(np.where(np.argmax(self.train_data.labels, axis=1) == c_)[0])
                    avg_sim = 0.0
                    max_sim = 0.0
                    min_sim = 0.0
                    max_idx = 0
                    min_idx = 0
                    #Iterate over all training points for current training class
                    for train_idx in train_class_indices:
                        train_image = self.train_data.images[train_idx]                        
                        train_label = self.train_data.labels[train_idx]
                    
                        feed_dict = {self.input_placeholder: train_image.reshape(1,-1), self.labels_placeholder: train_label.reshape(-1)}
                        grad_loss_wrt_train = np.concatenate(self.sess.run(self.grad_loss_wrt_param, feed_dict=feed_dict))
                        curr_sim = np.dot(grad_loss_wrt_train, grad_loss_wrt_test)
                        
                        if curr_sim > max_sim:
                            max_sim = curr_sim
                            max_idx = train_idx
                        if curr_sim < min_sim:
                            min_sim = curr_sim
                            min_idx = train_idx
                               
                        avg_sim += curr_sim
                
                    #Update similarity of train class with test point
                    sim_with_test_class[c_dx] += avg_sim / samples_per_class[c_]
                    
                    if verbose:  
                        print ('Test Idx: %d, Test Label: %d, Test Pred: %d, Train Class: %d, Avg Sim: %.8f, Max Sim: %.8f, Max Sim Idx: %d, Min Sim: %.8f, Min Idx: %d' % (test_idx, test_lb, test_pred, c_, sim_with_test_class[c_dx], max_sim, max_idx, min_sim, min_idx))

            #Divide similarity with number of test samples to get average similarity on test class
            sim_with_test_class /= float(num_test_samples)
            
            #Update influence matrix
            similarity_matrix[cdx,:] = sim_with_test_class[:]
    

        #Save results
        similarity_matrix.tofile(save_path)
        return similarity_matrix
    
    def evaluate_similarity_matrix_mod(self, test_classes=[], train_classes=[], num_test_samples=10, use_adv_sample=True, use_adv_label=False, save_path='mod_sim_matrix.dat', verbose=True):
        """
            Desc:
                Calculate similarity between gradient of loss wrt of test_classes and gradient of loss wrt train classes
            Params:
              test_classes(list): List containing test classses for which to calculate influence for
              train_classes(list): List containing training classes used for calculating infuence on each element of test_classes
              num_test_samples(int): number of test samples to use per class
              use_adv_sample(bool): Convert samples to adversarial using FGSM if true
              use_adv_label(bool): Treat network prediction on adversarial sample as the True label while calculating influence
              
          @return:
              similarity_matrix: matrix of size test_classes x train_classes 
        """

        
        train_classes = range(0,self.num_classes)
        test_classes = range(0,self.num_classes)
        similarity_matrix = np.zeros((len(test_classes), len(train_classes), len(test_classes) ))
        
       
        #Get number of samples per class
        samples_per_class = dict()

        #Stores number of samples per class in order to average over influence
        for c_ in train_classes:
            total_c_samples = np.sum(np.argmax(self.train_data.labels, axis=1) == c_)
            samples_per_class[c_] = float(total_c_samples)

        #Perform calculation for every class
        for cdx, c in enumerate(test_classes):
            #Get all data points from class and pick a random subset of test points
            data_points = np.where(np.argmax(self.test_data.labels, axis=1) == c)[0]
            test_class_indices = np.random.choice(data_points, num_test_samples)
        
            #Store results of influence for this test class
            sim_with_test_class = np.zeros((len(train_classes), len(train_classes)))
        
            #Iterate over test points for the current class 
            for test_idx in test_class_indices:
                test_image = self.test_data.images[test_idx]
                test_label = self.test_data.labels[test_idx]
                
                #Convert to adversarial sample
                if use_adv_sample == True:
                    #test_image = self.get_adversarial_version(test_image, test_label)
                    gt = np.argmax(test_label)
                    for cc in range(10):
                        test_label[:] = 0.0
                        test_label[cc] = 1.0
                        feed_dict = {self.input_placeholder: test_image.reshape(1,-1), self.labels_placeholder: test_label.reshape(-1)}
                        test_pred = self.sess.run(self.preds, feed_dict=feed_dict)[0]
                
                
                        test_pred = np.argmax(test_pred)
                        test_lb = np.argmax(test_label)
            
                        #Calculate inverse hvp
                        grad_loss_wrt_test = np.concatenate(self.sess.run(self.grad_loss_wrt_param , feed_dict=feed_dict))

                        #Evaluate the gradient vectors for all training classes in question
                        for c_dx, c_ in enumerate(train_classes):
                            train_class_indices = list(np.where(np.argmax(self.train_data.labels, axis=1) == c_)[0])
                            avg_sim = 0.0
                            max_sim = 0.0
                            min_sim = 0.0
                            max_idx = 0
                            min_idx = 0
                            #Iterate over all training points for current training class
                            for train_idx in train_class_indices:
                                train_image = self.train_data.images[train_idx]                        
                                train_label = self.train_data.labels[train_idx]
                    
                                feed_dict = {self.input_placeholder: train_image.reshape(1,-1), self.labels_placeholder: train_label.reshape(-1)}
                                grad_loss_wrt_train = np.concatenate(self.sess.run(self.grad_loss_wrt_param, feed_dict=feed_dict))
                                sim = np.dot(grad_loss_wrt_train, grad_loss_wrt_test)
                                
                        
                                if sim > max_sim:
                                    max_sim = sim
                                    max_idx = train_idx
                                if sim < min_sim:
                                    min_sim = sim
                                    min_idx = train_idx
                               
                                avg_sim += sim
                
                            #Update similarity of train class with test point
                            sim_with_test_class[cc, c_dx] += avg_sim / samples_per_class[c_]
                    
                            if verbose:  
                                print ('Test Class: %d, Test Label: %d, Test Pred: %d, Train Class: %d, Grad Class: %d, Avg Sim: %.8f, Max Sim: %.8f, Max Sim Idx: %d, Min Sim: %.8f, Min Idx: %d' % (test_idx, test_lb, test_pred, c_, cc, sim_with_test_class[cc, c_dx], max_sim, max_idx, min_sim, min_idx))

            #Divide similarity with number of test samples to get average similarity on test class
            sim_with_test_class /= float(num_test_samples)
            
            #Update influence matrix
            similarity_matrix[cdx,:, :] = sim_with_test_class
    

        #Save results
        similarity_matrix.tofile(save_path)
        return similarity_matrix

    def evaluate_influence_matrix_mod(self, test_classes=[], train_classes=[], num_test_samples=10, use_adv_sample=True, use_adv_label=False, save_path='mod_inf_matrix.dat', verbose=True):
        """
            Desc:
                Calculate similarity between gradient of loss wrt of test_classes and gradient of loss wrt train classes
            Params:
              test_classes(list): List containing test classses for which to calculate influence for
              train_classes(list): List containing training classes used for calculating infuence on each element of test_classes
              num_test_samples(int): number of test samples to use per class
              use_adv_sample(bool): Convert samples to adversarial using FGSM if true
              use_adv_label(bool): Treat network prediction on adversarial sample as the True label while calculating influence
              
          @return:
              similarity_matrix: matrix of size test_classes x train_classes 
        """

        
        train_classes = range(0,self.num_classes)
        test_classes = range(0,self.num_classes)
        influence_matrix = np.zeros((len(test_classes), len(train_classes), len(test_classes) ))
        
       
        #Get number of samples per class
        samples_per_class = dict()

        #Stores number of samples per class in order to average over influence
        for c_ in train_classes:
            total_c_samples = np.sum(np.argmax(self.train_data.labels, axis=1) == c_)
            samples_per_class[c_] = float(total_c_samples)

        #Perform calculation for every class
        for cdx, c in enumerate(test_classes):
            #Get all data points from class and pick a random subset of test points
            data_points = np.where(np.argmax(self.test_data.labels, axis=1) == c)[0]
            test_class_indices = np.random.choice(data_points, num_test_samples)
        
            #Store results of influence for this test class
            inf_with_test_class = np.zeros((len(train_classes), len(train_classes)))
        
            #Iterate over test points for the current class 
            for test_idx in test_class_indices:
                test_image = self.test_data.images[test_idx]
                test_label = self.test_data.labels[test_idx]
                
                #Convert to adversarial sample
                if use_adv_sample == True:
                    test_image = self.get_adversarial_version(test_image, test_label)
                    gt = np.argmax(test_label)
                    for cc in range(10):
                        test_label[:] = 0.0
                        test_label[cc] = 1.0
                        feed_dict = {self.input_placeholder: test_image.reshape(1,-1), self.labels_placeholder: test_label.reshape(-1)}
                        test_pred = self.sess.run(self.preds, feed_dict=feed_dict)[0]
                
                
                        test_pred = np.argmax(test_pred)
                        test_lb = np.argmax(test_label)
            
                        #Calculate inverse hvp
                        inverse_hvp = self.get_inverse_hvp(test_image, test_label)

                        #Evaluate the gradient vectors for all training classes in question
                        for c_dx, c_ in enumerate(train_classes):
                            train_class_indices = list(np.where(np.argmax(self.train_data.labels, axis=1) == c_)[0])
                            avg_inf = 0.0
                            max_inf = 0.0
                            min_inf = 0.0
                            max_idx = 0
                            min_idx = 0
                            #Iterate over all training points for current training class
                            for train_idx in train_class_indices:
                                train_image = self.train_data.images[train_idx]                        
                                train_label = self.train_data.labels[train_idx]
                    
                                feed_dict = {self.input_placeholder: train_image.reshape(1,-1), self.labels_placeholder: train_label.reshape(-1)}
                                inf = self.get_influence_on_test_loss(train_image, train_label, inverse_hvp)
                                
                        
                                if inf > max_inf:
                                    max_inf = inf
                                    max_idx = train_idx
                                if inf < min_inf:
                                    min_inf = inf
                                    min_idx = train_idx
                               
                                avg_inf += inf
                
                            #Update similarity of train class with test point
                            inf_with_test_class[cc, c_dx] += avg_inf / samples_per_class[c_]
                    
                            if verbose:  
                                print ('Test Class: %d, Test Label: %d, Test Pred: %d, Train Class: %d, Grad Class: %d, Avg Inf: %.8f, Max Inf: %.8f, Max Inf Idx: %d, Min Inf: %.8f, Min Idx: %d' % (test_idx, test_lb, test_pred, c_, cc, inf_with_test_class[cc, c_dx], max_inf, max_idx, min_inf, min_idx))

            #Divide similarity with number of test samples to get average similarity on test class
            inf_with_test_class /= float(num_test_samples)
            
            #Update influence matrix
            influence_matrix[cdx,:, :] = inf_with_test_class
    

        #Save results
        influence_matrix.tofile(save_path)
        return influence_matrix
    
    
    
    
    
    
