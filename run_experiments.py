from __future__ import division
from __future__ import print_function

import sys, os
sys.path.append('./src/')

import tensorflow as tf
import numpy as np

from influence.neural_network import NeuralNetwork
from influence.cnn import CNN
import influence.util as util

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#Seed used for all calculations of training and test point indices 
SEED = 14
#exp_number used to save all results to disk
exp_number = 'exp' + '2'

#Measure start time
import time
start_time = time.time()

#Load model from disk
model_name = 'CNN-MNIST-keras-1'
model_save_path = './trained_models/' + model_name + '-model.json'
weights_save_path = './trained_models/' + model_name + 'weights'
model = CNN(model_name=model_name)
epochs=10
model.train(epochs=epochs)
model.save_model(model_save_path, weights_save_path)    

#Generate test points and get test predictions
test_indices = model.gen_rand_indices_all_classes(y=model.test_labels, seed=SEED)

#Get Noisy, FGSM, and CW test points
test_noisy_data = model.generate_perturbed_data(model.test_data[test_indices], model.test_labels[test_indices],seed=SEED, perturbation='Noisy')
test_FGSM_data = model.generate_perturbed_data(model.test_data[test_indices], model.test_labels[test_indices],seed=SEED, perturbation='FGSM')
test_CW_data = model.generate_perturbed_data(model.test_data[test_indices], model.test_labels[test_indices],seed=SEED, perturbation='CW')

#Reset tf.graph() as Cleverhans modifies the graph
tf.reset_default_graph()

#Reload the model and weights
model = CNN(model_name=model_name)
model.load_model(model_save_path, weights_save_path)    

#Check dimensions of test data
print (test_noisy_data.shape)
print (test_FGSM_data.shape)
print (test_CW_data.shape)

#Get predictions for all the test data
test_noisy_preds = model.model.predict(test_noisy_data)
test_FGSM_preds = model.model.predict(test_FGSM_data)
test_CW_preds = model.model.predict(test_CW_data)
test_reg_preds = model.model.predict(model.test_data[test_indices])


#Test predictions shape
print (test_noisy_preds.shape)
print (test_FGSM_preds.shape)
print (test_CW_preds.shape)
print (test_reg_preds.shape)

#Convert predictions into one-hot-encodings
test_reg_preds_label = np.zeros_like(test_reg_preds)
test_reg_preds_label[range(0,test_reg_preds.shape[0]),np.argmax(test_reg_preds, axis=1)] = 1.0

test_noisy_preds_label = np.zeros_like(test_noisy_preds)
test_noisy_preds_label[range(0,test_noisy_preds.shape[0]),np.argmax(test_noisy_preds, axis=1)] = 1.0

test_FGSM_preds_label = np.zeros_like(test_FGSM_preds)
test_FGSM_preds_label[range(0,test_FGSM_preds.shape[0]),np.argmax(test_FGSM_preds, axis=1)] = 1.0

test_CW_preds_label = np.zeros_like(test_CW_preds)
test_CW_preds_label[range(0,test_CW_preds.shape[0]),np.argmax(test_CW_preds, axis=1)] = 1.0


#Save the data
matrix_save_path = './test_points/'
np.save(matrix_save_path + 'noisy_samples_test_' + exp_number, test_noisy_data)
np.save(matrix_save_path + 'fgsm_samples_test_' + exp_number, test_FGSM_data)
np.save(matrix_save_path + 'cw_samples_test_' + exp_number, test_CW_data)
np.save(matrix_save_path + 'reg_samples_test_' + exp_number, model.test_data[test_indices])


#Save the labels
np.save(matrix_save_path + 'noisy_labels_test_' + exp_number, test_noisy_preds_label)
np.save(matrix_save_path + 'fgsm_labels_test_' + exp_number, test_FGSM_preds_label)
np.save(matrix_save_path + 'cw_labels_test_' + exp_number, test_CW_preds_label)
np.save(matrix_save_path + 'reg_labels_test_' + exp_number, test_reg_preds_label)

#Get training sample indices for 1k, 10k, and 20k training points 
num_train_samples = 1000
data_indices_1k = model.gen_rand_indices(low=0, high=model.train_data.shape[0], seed=SEED, num_samples=num_train_samples)
train_data_1k = model.train_data[data_indices_1k]
train_labels_1k = model.train_labels[data_indices_1k]

num_train_samples = 10000
data_indices_10k = model.gen_rand_indices(low=0, high=model.train_data.shape[0], seed=SEED, num_samples=num_train_samples)
train_data_10k = model.train_data[data_indices_10k]
train_labels_10k = model.train_labels[data_indices_10k]

#WE do not do 20k training points due to IO space errors on the box
#num_train_samples = 20000
#data_indices_20k = model.gen_rand_indices(low=0, high=model.train_data.shape[0], seed=SEED, num_samples=num_train_samples)
#train_data_20k = model.train_data[data_indices_20k]
#train_labels_20k = np.argmax(model.train_labels[data_indices_20k], axis=1)

train_grad_save_path = './train_grads/train_grad_'

#Generate gradients for training points and save them to disk 
train_grads_1k = model.get_gradients_wrt_params(train_data_1k, train_labels_1k)
train_grads_10k = model.get_gradients_wrt_params(train_data_10k, train_labels_10k)

#Save gradients to disk
np.save(train_grad_save_path +'1k_' + exp_number + '.dat', train_grads_1k)
np.save(train_grad_save_path +'10k_' + exp_number + '.dat', train_grads_10k)

#Evaluate similarity matrices and save to disk
sim_matrix_1k_reg, _, _ = model.evaluate_similarity_matrix(sim_save_path='similarity_matrices/reg_sim_1k_' + exp_number + '.dat.npy', 
                                           use_train_grad_matrix=True,
                                           train_grad_matrix_path=train_grad_save_path +'1k_' + exp_number + '.dat.npy', 
                                           test_grad_save_path='test_grads/reg_test_grad_' + exp_number + '.dat.npy' , 
                                           test_points_path=matrix_save_path + 'reg_samples_test_' + exp_number + '.npy', 
                                           test_labels_path=matrix_save_path + 'reg_labels_test_' + exp_number + '.npy')

sim_matrix_1k_noisy, _, _ = model.evaluate_similarity_matrix(sim_save_path='similarity_matrices/noisy_sim_1k_' + exp_number + '.dat.npy', 
                                           use_train_grad_matrix=True,
                                           train_grad_matrix_path=train_grad_save_path +'1k_' + exp_number + '.dat.npy', 
                                           test_grad_save_path='test_grads/noisy_test_grad_' + exp_number + '.dat.npy' , 
                                           test_points_path=matrix_save_path + 'noisy_samples_test_' + exp_number + '.npy', 
                                           test_labels_path=matrix_save_path + 'noisy_labels_test_' + exp_number + '.npy')

sim_matrix_1k_fgsm, _, _ = model.evaluate_similarity_matrix(sim_save_path='similarity_matrices/fgsm_sim_1k_' + exp_number + '.dat.npy', 
                                           use_train_grad_matrix=True,
                                           train_grad_matrix_path=train_grad_save_path +'1k_' + exp_number + '.dat.npy', 
                                           test_grad_save_path='test_grads/fgsm_test_grad_' + exp_number + '.dat.npy' , 
                                           test_points_path=matrix_save_path + 'fgsm_samples_test_' + exp_number + '.npy', 
                                           test_labels_path=matrix_save_path + 'fgsm_labels_test_' + exp_number + '.npy')

sim_matrix_1k_cw, _, _ = model.evaluate_similarity_matrix(sim_save_path='similarity_matrices/cw_sim_1k_' + exp_number + '.dat.npy', 
                                           use_train_grad_matrix=True,
                                           train_grad_matrix_path=train_grad_save_path +'1k_' + exp_number + '.dat.npy', 
                                           test_grad_save_path='test_grads/cw_test_grad_' + exp_number + '.dat.npy' , 
                                           test_points_path=matrix_save_path + 'cw_samples_test_' + exp_number + '.npy', 
                                           test_labels_path=matrix_save_path + 'cw_labels_test_' + exp_number + '.npy')


sim_matrix_10k_reg, _, _ = model.evaluate_similarity_matrix(sim_save_path='similarity_matrices/reg_sim_10k_' + exp_number + '.dat.npy', 
                                           use_train_grad_matrix=True,
                                           use_test_grad_matrix=True,
                                           train_grad_matrix_path=train_grad_save_path +'10k_' + exp_number + '.dat.npy', 
                                           test_grad_matrix_path='test_grads/reg_test_grad_' + exp_number + '.dat.npy' , 
                                           test_points_path=matrix_save_path + 'reg_samples_test_' + exp_number + '.npy', 
                                           test_labels_path=matrix_save_path + 'reg_labels_test_' + exp_number + '.npy')

sim_matrix_10k_noisy, _, _ = model.evaluate_similarity_matrix(sim_save_path='similarity_matrices/noisy_sim_10k_' + exp_number + '.dat.npy', 
                                           use_train_grad_matrix=True,
                                           use_test_grad_matrix=True,
                                           train_grad_matrix_path=train_grad_save_path +'10k_' + exp_number + '.dat.npy', 
                                           test_grad_matrix_path='test_grads/noisy_test_grad_' + exp_number + '.dat.npy' , 
                                           test_points_path=matrix_save_path + 'noisy_samples_test_' + exp_number + '.npy', 
                                           test_labels_path=matrix_save_path + 'noisy_labels_test_' + exp_number + '.npy')

sim_matrix_10k_fgsm, _, _ = model.evaluate_similarity_matrix(sim_save_path='similarity_matrices/fgsm_sim_10k_' + exp_number + '.dat.npy', 
                                           use_train_grad_matrix=True,
                                           use_test_grad_matrix=True,
                                           train_grad_matrix_path=train_grad_save_path +'10k_' + exp_number + '.dat.npy', 
                                           test_grad_matrix_path='test_grads/fgsm_test_grad_' + exp_number + '.dat.npy' , 
                                           test_points_path=matrix_save_path + 'fgsm_samples_test_' + exp_number + '.npy', 
                                           test_labels_path=matrix_save_path + 'fgsm_labels_test_' + exp_number + '.npy')

sim_matrix_10k_cw, _, _ = model.evaluate_similarity_matrix(sim_save_path='similarity_matrices/cw_sim_10k_' + exp_number + '.dat.npy', 
                                           use_train_grad_matrix=True,
                                           use_test_grad_matrix=True,
                                           train_grad_matrix_path=train_grad_save_path +'10k_' + exp_number + '.dat.npy', 
                                           test_grad_matrix_path='test_grads/cw_test_grad_' + exp_number + '.dat.npy' , 
                                           test_points_path=matrix_save_path + 'cw_samples_test_' + exp_number + '.npy', 
                                           test_labels_path=matrix_save_path + 'cw_labels_test_' + exp_number + '.npy')


#Evaluate influence matrices
inf_matrix_1k_reg, _, _ = model.evaluate_influence_matrix(inf_save_path='influence_matrices/reg_inf_1k_' + exp_number + '.dat.npy', 
                                           use_grad_matrix=True,
                                           grad_matrix_path=train_grad_save_path +'1k_' + exp_number + '.dat.npy', 
                                           hvp_save_path='influence_matrices/reg_hvp_' + exp_number + '.dat.npy' , 
                                           test_points_path=matrix_save_path + 'reg_samples_test_' + exp_number + '.npy', 
                                           test_labels_path=matrix_save_path + 'reg_labels_test_' + exp_number + '.npy')

inf_matrix_1k_noisy, _, _ = model.evaluate_influence_matrix(inf_save_path='influence_matrices/noisy_inf_1k_' + exp_number + '.dat.npy', 
                                           use_grad_matrix=True,
                                           grad_matrix_path=train_grad_save_path +'1k_' + exp_number + '.dat.npy', 
                                           hvp_save_path='influence_matrices/noisy_hvp_' + exp_number + '.dat.npy' , 
                                           test_points_path=matrix_save_path + 'noisy_samples_test_' + exp_number + '.npy', 
                                           test_labels_path=matrix_save_path + 'noisy_labels_test_' + exp_number + '.npy')


inf_matrix_1k_fgsm, _, _ = model.evaluate_influence_matrix(inf_save_path='influence_matrices/fgsm_inf_1k_' + exp_number + '.dat.npy', 
                                           use_grad_matrix=True,
                                           grad_matrix_path=train_grad_save_path +'1k_' + exp_number + '.dat.npy', 
                                           hvp_save_path='influence_matrices/fgsm_hvp_' + exp_number + '.dat.npy' , 
                                           test_points_path=matrix_save_path + 'fgsm_samples_test_' + exp_number + '.npy', 
                                           test_labels_path=matrix_save_path + 'fgsm_labels_test_' + exp_number + '.npy')


inf_matrix_1k_cw, _, _ = model.evaluate_influence_matrix(inf_save_path='influence_matrices/cw_inf_1k_' + exp_number + '.dat.npy', 
                                           use_grad_matrix=True,
                                           grad_matrix_path=train_grad_save_path +'1k_' + exp_number + '.dat.npy', 
                                           hvp_save_path='influence_matrices/cw_hvp_' + exp_number + '.dat.npy' , 
                                           test_points_path=matrix_save_path + 'cw_samples_test_' + exp_number + '.npy', 
                                           test_labels_path=matrix_save_path + 'cw_labels_test_' + exp_number + '.npy')

inf_matrix_10k_reg, _, _ = model.evaluate_influence_matrix(inf_save_path='influence_matrices/reg_inf_10k_' + exp_number + '.dat.npy', 
                                           use_grad_matrix=True,
                                           use_hvp_matrix=True,
                                           grad_matrix_path=train_grad_save_path +'10k_' + exp_number + '.dat.npy', 
                                           hvp_matrix_path='influence_matrices/reg_hvp_' + exp_number + '.dat.npy' , 
                                           test_points_path=matrix_save_path + 'reg_samples_test_' + exp_number + '.npy', 
                                           test_labels_path=matrix_save_path + 'reg_labels_test_' + exp_number + '.npy')

inf_matrix_10k_noisy, _, _ = model.evaluate_influence_matrix(inf_save_path='influence_matrices/noisy_inf_10k_' + exp_number + '.dat.npy', 
                                           use_grad_matrix=True,
                                           use_hvp_matrix=True,
                                           grad_matrix_path=train_grad_save_path +'10k_' + exp_number + '.dat.npy', 
                                           hvp_matrix_path='influence_matrices/noisy_hvp_' + exp_number + '.dat.npy' , 
                                           test_points_path=matrix_save_path + 'noisy_samples_test_' + exp_number + '.npy', 
                                           test_labels_path=matrix_save_path + 'noisy_labels_test_' + exp_number + '.npy')

inf_matrix_10k_fgsm, _, _ = model.evaluate_influence_matrix(inf_save_path='influence_matrices/fgsm_inf_10k_' + exp_number + '.dat.npy', 
                                           use_grad_matrix=True,
                                           use_hvp_matrix=True,
                                           grad_matrix_path=train_grad_save_path +'10k_' + exp_number + '.dat.npy', 
                                           hvp_matrix_path='influence_matrices/fgsm_hvp_' + exp_number + '.dat.npy' , 
                                           test_points_path=matrix_save_path + 'fgsm_samples_test_' + exp_number + '.npy', 
                                           test_labels_path=matrix_save_path + 'fgsm_labels_test_' + exp_number + '.npy')

inf_matrix_10k_cw, _, _ = model.evaluate_influence_matrix(inf_save_path='influence_matrices/cw_inf_10k_' + exp_number + '.dat.npy', 
                                           use_grad_matrix=True,
                                           use_hvp_matrix=True,
                                           grad_matrix_path=train_grad_save_path +'10k_' + exp_number + '.dat.npy', 
                                           hvp_matrix_path='influence_matrices/cw_hvp_' + exp_number + '.dat.npy' , 
                                           test_points_path=matrix_save_path + 'cw_samples_test_' + exp_number + '.npy', 
                                           test_labels_path=matrix_save_path + 'cw_labels_test_' + exp_number + '.npy')

#Update train labels for analysis
train_labels_1k = np.argmax(train_labels_1k, axis=1)
train_labels_10k = np.argmax(train_labels_10k, axis=1)

#Update predictions for analysis
reg_pred = np.argmax(test_reg_preds, axis=1)
noisy_pred = np.argmax(test_noisy_preds, axis=1)
fgsm_pred = np.argmax(test_FGSM_preds, axis=1)
cw_pred = np.argmax(test_CW_preds, axis=1)

#Use verbose=True for detailed output
print ('REGULAR SIM ANALYSIS: ')
util.analyze_matrix(sim_matrix_1k_reg, reg_pred, train_labels=train_labels_1k, verbose=False)
util.analyze_matrix(sim_matrix_10k_reg, reg_pred, train_labels=train_labels_10k)

print ('\nNOISY SIM ANALYSIS: ')
util.analyze_matrix(sim_matrix_1k_noisy, noisy_pred, train_labels=train_labels_1k)
util.analyze_matrix(sim_matrix_10k_noisy, noisy_pred, train_labels=train_labels_10k)

print ('\nFGSM SIM ANALYSIS: ')
util.analyze_matrix(sim_matrix_1k_fgsm, fgsm_pred, train_labels=train_labels_1k)
util.analyze_matrix(sim_matrix_10k_fgsm, fgsm_pred, train_labels=train_labels_10k)

print ('\nCW SIM ANALYSIS: ')
util.analyze_matrix(sim_matrix_1k_cw, cw_pred, train_labels=train_labels_1k)
util.analyze_matrix(sim_matrix_10k_cw, cw_pred, train_labels=train_labels_10k)



print ('REGULAR INF ANALYSIS: ')
util.analyze_matrix(inf_matrix_1k_reg, reg_pred, train_labels=train_labels_1k, verbose=False)
util.analyze_matrix(inf_matrix_10k_reg, reg_pred, train_labels=train_labels_10k)

print ('\nNOISY INF ANALYSIS: ')
util.analyze_matrix(inf_matrix_1k_noisy, noisy_pred, train_labels=train_labels_1k)
util.analyze_matrix(inf_matrix_10k_noisy, noisy_pred, train_labels=train_labels_10k)

print ('\nFGSM INF ANALYSIS: ')
util.analyze_matrix(inf_matrix_1k_fgsm, fgsm_pred, train_labels=train_labels_1k)
util.analyze_matrix(inf_matrix_10k_fgsm, fgsm_pred, train_labels=train_labels_10k)

print ('\nCW INF ANALYSIS: ')
util.analyze_matrix(inf_matrix_1k_cw, cw_pred, train_labels=train_labels_1k)
util.analyze_matrix(inf_matrix_10k_cw, cw_pred, train_labels=train_labels_10k)



print ('\n\nNOISY VS REG SIM: ')
print ('1k')
util.compare_matrices(sim_matrix_1k_noisy, sim_matrix_1k_reg, noisy_pred, reg_pred, train_labels=train_labels_1k)
print ('10k')
util.compare_matrices(sim_matrix_10k_noisy, sim_matrix_10k_reg, noisy_pred, reg_pred, train_labels=train_labels_1k)


print ('\n\nFGSM VS REG SIM: ')
print ('1k')
util.compare_matrices(sim_matrix_1k_fgsm, sim_matrix_1k_reg, fgsm_pred, reg_pred, train_labels=train_labels_1k)
print ('10k')
util.compare_matrices(sim_matrix_10k_fgsm, sim_matrix_10k_reg, fgsm_pred, reg_pred, train_labels=train_labels_1k)

print ('\n\nCW VS REG SIM: ')
print ('1k')
util.compare_matrices(sim_matrix_1k_cw, sim_matrix_1k_reg, cw_pred, reg_pred, train_labels=train_labels_1k)
print ('10k')
util.compare_matrices(sim_matrix_10k_cw, sim_matrix_10k_reg, cw_pred, reg_pred, train_labels=train_labels_1k)

print ('\n\nFGSM VS NOISY SIM: ')
print ('1k')
util.compare_matrices(sim_matrix_1k_fgsm, sim_matrix_1k_noisy, fgsm_pred, noisy_pred, train_labels=train_labels_1k)
print ('10k')
util.compare_matrices(sim_matrix_10k_fgsm, sim_matrix_10k_noisy, fgsm_pred, noisy_pred, train_labels=train_labels_1k)


print ('\n\nCW VS FGSM SIM: ')
print ('1k')
util.compare_matrices(sim_matrix_1k_fgsm, sim_matrix_1k_fgsm, cw_pred, fgsm_pred, train_labels=train_labels_1k)
print ('10k')
util.compare_matrices(sim_matrix_10k_cw, sim_matrix_10k_fgsm, cw_pred, fgsm_pred, train_labels=train_labels_1k)


print ('\n\nNOISY VS REG INF: ')
print ('1k')
util.compare_matrices(inf_matrix_1k_noisy, inf_matrix_1k_reg, noisy_pred, reg_pred, train_labels=train_labels_1k)
print ('10k')
util.compare_matrices(inf_matrix_10k_noisy, inf_matrix_10k_reg, noisy_pred, reg_pred, train_labels=train_labels_1k)


print ('\n\nFGSM VS REG SIM: ')
print ('1k')
util.compare_matrices(inf_matrix_1k_fgsm, inf_matrix_1k_reg, fgsm_pred, reg_pred, train_labels=train_labels_1k)
print ('10k')
util.compare_matrices(inf_matrix_10k_fgsm, inf_matrix_10k_reg, fgsm_pred, reg_pred, train_labels=train_labels_1k)

print ('\n\nCW VS REG SIM: ')
print ('1k')
util.compare_matrices(inf_matrix_1k_cw, inf_matrix_1k_reg, cw_pred, reg_pred, train_labels=train_labels_1k)
print ('10k')
util.compare_matrices(inf_matrix_10k_cw, inf_matrix_10k_reg, cw_pred, reg_pred, train_labels=train_labels_1k)

print ('\n\nFGSM VS NOISY SIM: ')
print ('1k')
util.compare_matrices(inf_matrix_1k_fgsm, inf_matrix_1k_noisy, fgsm_pred, noisy_pred, train_labels=train_labels_1k)
print ('10k')
util.compare_matrices(inf_matrix_10k_fgsm, inf_matrix_10k_noisy, fgsm_pred, noisy_pred, train_labels=train_labels_1k)


print ('\n\nCW VS FGSM SIM: ')
print ('1k')
util.compare_matrices(inf_matrix_1k_fgsm, inf_matrix_1k_fgsm, cw_pred, fgsm_pred, train_labels=train_labels_1k)
print ('10k')
util.compare_matrices(inf_matrix_10k_cw, inf_matrix_10k_fgsm, cw_pred, fgsm_pred, train_labels=train_labels_1k)




