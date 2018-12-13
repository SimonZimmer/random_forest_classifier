from essentia import *
from essentia.standard import *
import pandas
import os
import numpy as np
import pickle

list_of_files = os.listdir("./audio_files/")
num_files = list_of_files.__len__()

print('preparing extractor...')
extractor = FreesoundExtractor(profile='custom_extractor_profile.yaml')
features = extractor("./audio_files/" + list_of_files[0])
features = features[0]
output = pandas.DataFrame(index=range(num_files), columns=[features.descriptorNames()], dtype='float')
all_features = features.descriptorNames()

print('extracting features...')
for file in range(num_files):
    if list_of_files[file] == '.DS_Store':
        print('DS_Store item encountered & removed')
        os.remove(list_of_files[file])
    path = "./audio_files/" + list_of_files[file]
    features = extractor(path)
    features = features[0]

    for feat in range(len(all_features)):
        current_feat = str(all_features[feat])
        if isinstance(features[current_feat], float):
            output.iloc[file, feat] = features[str(all_features[feat])]
        else:
            break
    begin_list = feat + 1
    for feat in range(begin_list, len(all_features)):
        current_feature = features[str(all_features[feat])]
        if (type(current_feature) != str) or (not current_feature_mean):
            current_feature_list = current_feature.tolist()
            current_feature_mean = np.mean(current_feature_list)
            if np.isnan(current_feature_mean):
                output.iloc[file, feat] = 0.0
            else:
                output.iloc[file, feat] = current_feature_mean
        else: break
    end_list = feat
    print('extracting file ' + str(file) + ' of ' + str(num_files))

output.dropna(axis=1, inplace = True)

print('annotate data...')
values = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]
target = pandas.Series(data=values, index=range(0,len(values)), dtype=int, name='target')
output = pandas.concat([output, target], axis=1)

output.to_pickle('test_files_features.pkl')
print(output)