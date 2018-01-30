from __future__ import division
from __future__ import print_function

import sys, os
sys.path.append('./src/')

import tensorflow as tf
from influence.neural_network import NeuralNetwork
from influence.cnn import CNN
import influence.util as util


def get_similarity(option=0, device="0", seed = 12, model_name="CNN-MNIST"):
  """
  Desc: Calculate similarity of test points wrt randomly chosen test points
  """
  os.environ["CUDA_VISIBLE_DEVICES"] = device
  SEED = seed

  model_name = model_name
  if option == 0:
    ext = '_10x10_reg_sample_and_label'
  elif option == 1:
    ext = '_10x10_adv_sample_and_label'
  else:
    ext = '_10x10_adv_sample'

  save_path = './similarity_matrices/' + model_name + ext


  tf.reset_default_graph()
  g = tf.Graph()
  with g.as_default():
    tf.set_random_seed(SEED)
    model = CNN(model_name=model_name, load_from_file=True, model_chkpt_file='./trained_models/' + model_name + '.ckpt')
    if option == 0:
      sim = model.evaluate_similarity_matrix(save_path=save_path)
    elif option == 1:
      sim = model.evaluate_similarity_matrix(use_adv_sample=True, use_adv_label=True,save_path=save_path)
    else:
      sim = model.evaluate_similarity_matrix(use_adv_sample=True, save_path=save_path)
    
    print (sim)

def get_influence(option=0, device="0", seed = 12, model_name="CNN-MNIST"):
  """
  Desc: Calculate similarity of test points wrt randomly chosen test points
  """
  os.environ["CUDA_VISIBLE_DEVICES"] = device
  SEED = seed

  model_name = model_name
  if option == 0:
    ext = '_10x10_reg_sample_and_label'
  elif option == 1:
    ext = '_10x10_adv_sample_and_label'
  else:
    ext = '_10x10_adv_sample'

  save_path = './influence_matrices/' + model_name + ext


  tf.reset_default_graph()
  g = tf.Graph()
  with g.as_default():
    tf.set_random_seed(SEED)
    model = CNN(model_name=model_name, load_from_file=True, model_chkpt_file='./trained_models/' + model_name + '.ckpt')
    if option == 0:
      inf = model.evaluate_influence_matrix(save_path=save_path)
    elif option == 1:
      inf = model.evaluate_influence_matrix(use_adv_sample=True, use_adv_label=True,save_path=save_path)
    else:
      inf = model.evaluate_influence_matrix(use_adv_sample=True, save_path=save_path)
    
    print (inf)

if __name__ == '__main__':
  if len(sys.argv) > 3:
    if sys.argv[1] == "sim":
      get_similarity(option=int(sys.argv[2]), device =sys.argv[3])
    else:
      get_influence(option=int(sys.argv[2]), device=sys.argv[3])




