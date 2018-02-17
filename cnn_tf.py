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

from influence.neural_network import NeuralNetwork

SEED = 12


class CNN(NeuralNetwork):
    """Convolutional Neural Network - 2 hidden layers (for now) """

    def __init__(self, input_side=28, input_channels=1, non_linearity='tanh', conv1_ch=32, conv2_ch=32, **kwargs):
        """
        Params:
        input_side(int): size of the y-axis. We are assuming x = y 
        input_channels(int): number of input channels in the input
        non_linearity(str): can be 'tanh', 'sigmoid'
        conv[1,2]_channels(int): number of channels in the conv filter
       
        """
        self.input_side = input_side
        self.input_channels = input_channels
        self.input_dim = self.input_side * self.input_side * self.input_channels
        self.conv_patch_size = 5
        if non_linearity == 'tanh':
            self.non_linearity = tf.nn.tanh
        else:
            self.non_linearity = tf.sigmoid
        self.conv1_ch = conv1_ch
        self.conv2_ch = conv2_ch

        super(CNN, self).__init__(**kwargs)


    def get_params(self):
        """
        Desc:
            Required for getting params to be used in HVP Lissa calculations
        """
        all_params = []
        for layer in ['conv1', 'conv2', 'dense']:        
            for var_name in ['weights', 'biases']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))            
                all_params.append(temp_tensor)      
        return all_params        
        

    def create_input_placeholders(self):
        """
        Desc:
            Create input place holders for graph and model.
        """
        input_placeholder = tf.placeholder(
            tf.float32, 
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,             
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder
    
    
    def get_conv_layer(self, input_x, conv_patch_size, input_channels, output_channels, stride):
        """
        Desc:
            Create a Convolution layer in tensorflow with the specified attributes.
        """
        
        std_dev = 2.0 / math.sqrt(float(conv_patch_size * conv_patch_size * input_channels))
        initializer = tf.truncated_normal_initializer(stddev = std_dev, dtype=tf.float32)
                                        
        weights = tf.get_variable( 'weights', 
                                    [conv_patch_size*conv_patch_size*input_channels*output_channels], 
                                    tf.float32,
                                    initializer=initializer)
                                  
        biases = tf.get_variable('biases',
                                [output_channels],
                                tf.float32,
                                tf.constant_initializer(0.0))
                                  
        weights_reshaped = tf.reshape(weights, [conv_patch_size, conv_patch_size, input_channels, output_channels])
                                  
        layer = self.non_linearity(
                    tf.nn.conv2d(input_x, weights_reshaped, strides=[1,stride,stride,1], padding='VALID') + biases)
        
  
        return layer
    
    def get_dense_layer(self, input_x, input_size):
        """
        Desc:
            Create a fully connected layer. 
        """
                 
        std_dev = 1.0 / math.sqrt(float(input_size))
        initializer = tf.truncated_normal_initializer(stddev = std_dev, dtype=tf.float32)
                                        
        weights = tf.get_variable( 'weights', 
                                    [input_size * self.num_classes], 
                                    tf.float32,
                                    initializer=initializer)
                                  
        biases = tf.get_variable('biases',
                                [self.num_classes],
                                 tf.float32,
                                tf.constant_initializer(0.0))
                                  
        weights_reshaped = tf.reshape(weights, [input_size, self.num_classes])
                                  
        layer = tf.matmul(input_x, weights_reshaped) + biases
        
        return layer



    def forward_pass(self, input_x):  
        """
        Desc:
            Populate the tf graph with operations for forward pass through the network. 
        
        """

        #28 x 28
        input_reshaped = tf.reshape(input_x, [-1, self.input_side, self.input_side, self.input_channels])

        # Convolutional Layer #1
        with tf.variable_scope('conv1'):
            conv1 = self.get_conv_layer(input_reshaped, self.conv_patch_size, self.input_channels, self.conv1_ch,1)
        
        #with tf.variable_scope('pool1'):
        #    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 
        with tf.variable_scope('conv2'):
            conv2 = self.get_conv_layer(conv1, self.conv_patch_size, self.conv1_ch, self.conv2_ch,1)
        
        #with tf.variable_scope('pool2'):
        #    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
                                                   
        # Dense Layer
        conv2_flat = tf.reshape(conv2, [-1, 20 * 20 * self.conv2_ch])
        
        with tf.variable_scope('dropout'):
            dropout = tf.layers.dropout(inputs=conv2_flat, rate=self.dropout_prob)

                                        
        with tf.variable_scope('dense'):
            logits = self.get_dense_layer(dropout, 20*20*self.conv2_ch)
       
            
        return logits

    def predict(self, logits):
        """
        Desc:
            Apply softmax to network outputs
        """
        preds = tf.nn.softmax(logits, name='preds')
        return preds
    
    