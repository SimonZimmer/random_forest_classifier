
import essentia.standard
import pandas
import os
import numpy as np


input_path = "./datasets/vaillant_1/audio/"
list_of_files = os.listdir(input_path)
num_files = list_of_files.__len__()

print('preparing extractor...')

extractor = essentia.standard.FreesoundExtractor(profile='custom_extractor_profile.yaml')
features = extractor(input_path + list_of_files[0])
features = features[0]
output = pandas.DataFrame(index=range(num_files), columns=[features.descriptorNames()], dtype='float')
all_features = features.descriptorNames()

print('extracting features...')
for file in range(num_files):
    if list_of_files[file] == '.DS_Store':
        print('DS_Store item encountered & removed')
        os.remove(list_of_files[file])
    path = input_path + list_of_files[file]
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

output.dropna(axis=1, inplace=True)
output.to_pickle(os.getcwd() + "features.pkl")
print(output)
