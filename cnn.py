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
from keras import regularizers
from keras.regularizers import l2
from keras.models import Sequential, model_from_json
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D

from influence.neural_network import NeuralNetwork

SEED = 12


class CNN(NeuralNetwork):
    """Convolutional Neural Network - 2 hidden layers (for now) """

    def __init__(self, input_side=28, input_channels=1, non_linearity='sigmoid', conv1_ch=32, conv2_ch=64, **kwargs):
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
        self.non_linearity = non_linearity
        self.conv1_ch = conv1_ch
        self.conv2_ch = conv2_ch

        super(CNN, self).__init__(**kwargs)
        

    def create_model(self):
        """
        Create a Keras Sequential Model
        """
        layers = [
        Conv2D(
            self.conv1_ch, 
            (self.conv_patch_size, self.conv_patch_size),
            padding='valid',
            input_shape=(self.input_side, self.input_side, 1),
            name='conv1'
           ),
        Activation(self.non_linearity),
        Conv2D(self.conv2_ch, (self.conv_patch_size, self.conv_patch_size), name='conv2'),
        Activation(self.non_linearity),
        Dropout(self.dropout_prob),
        Flatten(),
        Dense(128, name='dense1'),
        Activation(self.non_linearity),
        Dropout(self.dropout_prob),
        Dense(self.num_classes, name='logits'),
        Activation('softmax')
        ]
    
    
        model = Sequential()
        for layer in layers:
            model.add(layer)
        return model
    
    def get_params(self):
        """
        Desc:
            Required for getting params to be used in HVP Lissa calculations
        """
        all_params = []
        for layer in self.model.layers:
            if layer.name in ['conv1', 'conv2', 'dense1', 'logits']:        
                for weight in layer.trainable_weights:
                    all_params.append(weight)  
        return all_params        
        

    def create_input_placeholders(self):
        """
        Desc:
            Create input place holders for graph and model.
        """
        input_placeholder = tf.placeholder(
            tf.float32, 
            shape=(None, self.input_dim, self.input_dim, 1),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,             
            shape=(None,self.num_classes),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder
    
    
    def get_logits_preds(self, inputs):
        """
        Desc:
            Get logits of models
        """
        preds = self.model(inputs)
        logits, = preds.op.inputs #inputs to the softmax operation
        return logits, preds
    
    def compile_model(self):
        """
        Initialize the model
        """
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
        
    def save_model(self, store_path_model, store_path_weights):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(store_path_model, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(store_path_weights)
        print("Saved model to disk")
 
    def load_model(self, load_path_model, load_path_weights):
        # load json and create model
        json_file = open(load_path_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(load_path_weights)
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
        print("Loaded model from disk")
        
    def reshape_data(self):
        """
        Desc:
            Trains model for a specified number of epochs.
        """    
        
        
        self.train_data = self.train_data.reshape(-1, 28, 28, 1)
        self.val_data = self.val_data.reshape(-1, 28, 28, 1)
        self.test_data = self.test_data.reshape(-1, 28, 28, 1)
    
        self.train_data = self.train_data.astype('float32')
        self.val_data = self.val_data.astype('float32')
        self.test_data = self.test_data.astype('float32')
        
        
        