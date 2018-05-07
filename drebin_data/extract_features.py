import os
import pickle
count = 0
feat_set=dict()
f_count = 0
dir_ = './feature_vectors'

for f in os.listdir(dir_):
  lines = [l for l in open(os.path.join(dir_,f),'r')]
  if len(lines) > 0:
    for l in lines:
      if len(l.split('::')) > 1:
        feat = l.split('::')
        feature = ""
        for fe in feat:
          feature = fe + feature
        #Replace newline
        feature = feature.replace('\n','')
        if feature not in feat_set:
          feat_set[feature] = f_count
          f_count+=1

print f_count
with open('./drebin_features.pkl', 'wb') as f:
        pickle.dump(feat_set, f, pickle.HIGHEST_PROTOCOL)

print len(feat_set)
