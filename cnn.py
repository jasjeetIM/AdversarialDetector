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
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from influence.neural_network import NeuralNetwork

SEED = 12


class CNN(NeuralNetwork):
    """Convolutional Neural Network - 2 hidden layers (for now) """

    def __init__(self, non_linearity='relu', **kwargs):
        """
        Params:
        input_side(int): size of the y-axis. We are assuming x = y 
        input_channels(int): number of input channels in the input
        non_linearity(str): can be 'tanh', 'sigmoid'
        conv[1,2]_channels(int): number of channels in the conv filter
       
        """
        
        self.non_linearity = non_linearity

        super(CNN, self).__init__(**kwargs)
        
    def create_model(self, dataset='mnist'):
        """
        Create a Keras Sequential Model
        """
        if dataset.lower() == 'mnist':
            layers = [
                Conv2D(32, (5, 5), padding='valid', input_shape=(self.input_side, self.input_side, self.input_channels), name='conv1'),
                Activation(self.non_linearity),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(64, (5, 5), name='conv2'),
                Activation(self.non_linearity),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(self.dropout_prob),
                Flatten(),
                Dense(128, name='dense1'),
                Activation(self.non_linearity),
                Dropout(self.dropout_prob),
                Dense(self.num_classes, name='logits'),
                Activation('softmax')
            ]
            
        if dataset.lower() == 'mnist-inf':
            layers = [
                Conv2D(32, (5, 5), padding='valid', input_shape=(self.input_side, self.input_side, self.input_channels), name='conv1'),
                Activation(self.non_linearity),
                Conv2D(64, (5, 5), name='conv2'),
                Activation(self.non_linearity),
                Dropout(self.dropout_prob),
                Flatten(),
                Dense(128, name='dense1'),
                Activation(self.non_linearity),
                Dropout(self.dropout_prob),
                Dense(self.num_classes, name='logits'),
                Activation('softmax')
            ]
    
        elif dataset.lower() == 'cifar10':
            layers = [
                Conv2D(32, (3, 3), padding='same', input_shape=(self.input_side, self.input_side, self.input_channels), name='conv1'),
                Activation(self.non_linearity),
                Conv2D(32, (3, 3), padding='same', name='conv2'),
                Activation(self.non_linearity),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(64, (3, 3), padding='same', name='conv3'),
                Activation(self.non_linearity),
                Conv2D(64, (3, 3), padding='same', name='conv4'),
                Activation(self.non_linearity),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(128, (3, 3), padding='same', name='conv5'),
                Activation(self.non_linearity),
                Conv2D(128, (3, 3), padding='same', name='conv6'),
                Activation(self.non_linearity),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dropout(self.dropout_prob),
                Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='dense1'),
                Activation(self.non_linearity),
                Dropout(self.dropout_prob),
                Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='dense2'),
                Activation(self.non_linearity),
                Dropout(self.dropout_prob),
                Dense(self.num_classes, name='logits'),
                Activation('softmax')
            ]
            
        elif dataset.lower() == 'cifar10-inf':
            layers = [
                Conv2D(32, (3, 3), padding='same', input_shape=(self.input_side, self.input_side, self.input_channels), name='conv1'),
                Activation(self.non_linearity),
                Conv2D(32, (3, 3), padding='same', name='conv2'),
                Activation(self.non_linearity),
                Conv2D(64, (3, 3), padding='same', name='conv3'),
                Activation(self.non_linearity),
                Conv2D(64, (3, 3), padding='same', name='conv4'),
                Activation(self.non_linearity),
                Conv2D(128, (3, 3), padding='same', name='conv5'),
                Activation(self.non_linearity),
                Conv2D(128, (3, 3), padding='same', name='conv6'),
                Activation(self.non_linearity),
                Flatten(),
                Dropout(self.dropout_prob),
                Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='dense1'),
                Activation(self.non_linearity),
                Dropout(self.dropout_prob),
                Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='dense2'),
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
            for weight in layer.trainable_weights:
                all_params.append(weight)
        return all_params        
        

    def create_input_placeholders(self):
        """
        Desc:
            Create input place holders for graph and model.
        """
        input_shape_all = (None,self.input_side, self.input_side,self.input_channels)
        label_shape_all = (None,self.num_classes)
        input_placeholder = tf.placeholder(
            tf.float32, 
            shape=input_shape_all,
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,             
            shape=label_shape_all,
            name='labels_placeholder')
        input_shape_one = (1,self.input_side, self.input_side,self.input_channels)
        label_shape_one = (1,self.num_classes)
        
        return input_placeholder, labels_placeholder, input_shape_one, label_shape_one
    
    
    def get_logits_preds(self, inputs):
        """
        Desc:
            Get logits of models
        """
        preds = self.model(inputs)
        logits, = preds.op.inputs #inputs to the softmax operation
        return logits, preds
    
    def predict(self, x):
        feed_dict = {
                self.input_placeholder: x.reshape(self.input_shape),
                K.learning_phase(): 0
            } 
        preds = self.sess.run(self.preds, feed_dict=feed_dict)
        return preds
        
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
        # load weights into new model
        self.model.load_weights(load_path_weights)
        print("Loaded model from disk")
        
    def reshape_data(self):
        """
        Desc:
            Reshapes data to original size
        """    
        self.train_data = self.train_data.reshape(-1, self.input_side, self.input_side, self.input_channels)
        self.val_data = self.val_data.reshape(-1, self.input_side, self.input_side, self.input_channels)
        self.test_data = self.test_data.reshape(-1, self.input_side, self.input_side, self.input_channels)
    
        self.train_data = self.train_data.astype('float32')
        self.val_data = self.val_data.astype('float32')
        self.test_data = self.test_data.astype('float32')
        
        
        