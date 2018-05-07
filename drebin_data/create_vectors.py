import pickle as p
import os
import numpy as np

dir_ = './feature_vectors'
mal_files = [l.split(',')[0] for l in open('malware_names.txt','r')]
feat_dict = p.load(open('./drebin_features.pkl','rb'))

dim = len(feat_dict)
num_total = 129013
num_mal = len(mal_files)
num_clean = num_total - num_mal

ben_matrix = np.zeros((num_clean, dim), dtype=np.int8)
mal_matrix = np.zeros((num_mal, dim), dtype=np.int8)

idx = 0
#Get clean data first 
for  f in os.listdir(dir_):
  if f not in mal_files:
    lines = [l for l in open(os.path.join(dir_,f), 'r')]
    for l in lines:
      if len(l.split('::')) > 1:
        feat = l.split('::')
        feature = ""
        for fe in feat:
          feature = fe + feature
        feature = feature.replace('\n','')
        ben_matrix[idx,feat_dict[feature]] = 1
    idx+=1
            

for idx, f in enumerate(mal_files):    
    lines = [l for l in open(os.path.join(dir_,f), 'r') ]
    for l in lines:
      if len(l.split('::')) > 1:
        feat = l.split('::')
        feature = ""
        for fe in feat:
          feature = fe + feature
        feature = feature.replace('\n','')
        mal_matrix[idx,feat_dict[feature]] = 1
    

np.save('mal_matrix', mal_matrix)
np.save('ben_matrix', ben_matrix)
