from __future__ import division
from __future__ import print_function

import sys, os
sys.path.append('./src')
sys.path.append('/root/notebook/cleverhans')

import tensorflow as tf
from influence.neural_network import NeuralNetwork
from influence.cnn import CNN
import influence.util as util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def get_similarity(option=0, device="0", seed = 14, model_name="CNN-MNIST-keras-1"):
  """
  Desc: Calculate similarity of test points wrt randomly chosen test points
  """
  os.environ["CUDA_VISIBLE_DEVICES"] = device
  SEED = seed

  model_name = model_name
  tf.set_random_seed(SEED)
  model = CNN(model_name=model_name)
  model.load_model('./trained_models/' + model_name + '-model.json', './trained_models/' + model_name + 'weights')
  
  if option == 0:
    sim_matrix, test_gradients, train_gradients = model.evaluate_similarity_matrix(sim_save_path='similarity_matrices/reg_sim_1k_exp2.dat.npy', train_grad_save_path='influence_matrices/train_grad_1k_exp2.dat.npy',test_grad_save_path='similarity_matrices/reg_test_grad_exp2.dat.npy' , test_points_path='reg_samples_test_exp2.npy', test_labels_path='reg_labels_test_exp2.npy',num_train_samples=1000, seed=SEED)
  elif option == 1:
    sim_matrix, test_gradients, train_gradients = model.evaluate_similarity_matrix(use_adv_sample=True, use_adv_label=True, sim_save_path='similarity_matrices/adv_sim_20k.dat.npy', use_train_grad_matrix=True, train_grad_matrix_path='influence_matrices/train_grad_20k.dat.npy',use_test_grad_matrix=True,test_grad_matrix_path='similarity_matrices/adv_test_grad.dat.npy')
  elif option == 2:
    sim_matrix, test_gradients, train_gradients = model.evaluate_similarity_matrix(use_rand_sample=True, use_rand_label=True,sim_save_path='similarity_matrices/rand_sim_20k.dat.npy', use_train_grad_matrix=True, train_grad_matrix_path='influence_matrices/train_grad_20k.dat.npy',use_test_grad_matrix=True, test_grad_matrix_path='similarity_matrices/rand_test_grad.dat.npy' )

def get_influence(option=0, device="0", seed = 14, model_name="CNN-MNIST-keras-1"):
  """
  Desc: Calculate similarity of test points wrt randomly chosen test points
  """
  os.environ["CUDA_VISIBLE_DEVICES"] = device
  SEED = seed

  model_name = model_name
  tf.set_random_seed(SEED)
  model = CNN(model_name=model_name)
  model.load_model('./trained_models/' + model_name + '-model.json', './trained_models/' + model_name + 'weights')
  
  if option == 0:
    inf_matrix, hvp_matrix, train_gradients = model.evaluate_influence_matrix(use_reg_label=True,inf_save_path='influence_matrices/reg_inf_20k.dat.npy', use_grad_matrix=True, grad_matrix_path='influence_matrices/train_grad_20k.dat.npy', use_hvp_matrix=True,hvp_matrix_path='influence_matrices/reg_hvp.dat.npy')
  elif option == 1:
    inf_matrix, hvp_matrix, train_gradients = model.evaluate_influence_matrix(use_adv_sample=True,use_adv_label=True,inf_save_path='influence_matrices/adv_inf_20k.dat.npy', use_grad_matrix=True, grad_matrix_path='influence_matrices/train_grad_20k.dat.npy', use_hvp_matrix=True, hvp_matrix_path='influence_matrices/adv_hvp.dat.npy')
  elif option == 2:
    inf_matrix, hvp_matrix, train_gradients = model.evaluate_influence_matrix(use_rand_sample=True,use_rand_label=True, inf_save_path='influence_matrices/rand_inf_20k.dat.npy', use_grad_matrix=True, grad_matrix_path='influence_matrices/train_grad_20k.dat.npy',use_hvp_matrix=True,hvp_matrix_path='influence_matrices/rand_hvp.dat.npy')


if __name__ == '__main__':
  if len(sys.argv) > 3:
    if sys.argv[1] == "sim":
      get_similarity(option=int(sys.argv[2]), device =sys.argv[3])
    else:
      get_influence(option=int(sys.argv[2]), device=sys.argv[3])




