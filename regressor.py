import pickle as pkl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns


label_path = "/Users/simonzimmermann/dev/random_forest_regressor/datasets/vaillant_1/pkl/vaillant_1.pkl"
print('load data...')
data = pkl.load(open("features.pkl"))

target_index = len(data.iloc[0]) - 1
x_input = data.iloc[:, : target_index - 1]
x_input = x_input.fillna(0)
target = pkl.load(open(label_path))

x_train, x_test, y_train, y_test = train_test_split(x_input, target, test_size=0.3)

print('training regressor...')
regressor = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print("score= " + str(regressor.score(x_test, y_test)))

print('plotting feature importance...')
feature_imp = pd.Series(regressor.feature_importances_, index=x_input.columns).sort_values(ascending=False)
feature_imp = feature_imp[feature_imp > 0]
print(feature_imp)

a4_dims = (10, 200)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barp
import essentia
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
output.to_pickle("features.pkl")
print(output)
lot(ax=ax, x=feature_imp, y=feature_imp.index)
plt.title("Feature Importance Score")
plt.ylabel('Features')
plt.title("Audiotory Feature Importance")
plt.legend()