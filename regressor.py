import pickle as pkl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns


label_path = "/Users/simonzimmermann/dev/random_forest_regressor/datasets/vaillant_1/pkl/vaillant_1.pkl"
print('load data...')
data = pkl.load(open("features.pkl"))

data = pkl.load(open("/Users/simonzimmermann/dev/random_forest_regressor/test_files_features.pkl"))

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
sns.barplot(ax=ax, x=feature_imp, y=feature_imp.index)
plt.title("Feature Importance Score")
plt.ylabel('Features')
plt.title("Audiotory Feature Importance")
plt.legend()