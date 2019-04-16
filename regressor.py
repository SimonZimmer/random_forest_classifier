import pickle as pkl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns


def run_regressor(label_path, score_list, plot):
    """run an instance of a random forest regressor.

    :param label_path: the path to a .pkl file containing the target annotations
    :param score_list: list for storing the score of the model
    :param plot: flag to signalise if a feature importance barplot should be created
    :return list with appended score of the model
    """
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
    print("score= " + str(regressor.score(x_test, y_test)))
    score_list.append(regressor.score(x_test, y_test))

    # make a barplot diagram to determine the most important features
    if plot:
        print('plotting feature importance...')
        feature_imp = pd.Series(regressor.feature_importances_, index=x_input.columns).sort_values(ascending=False)
        feature_imp = feature_imp[feature_imp > 0]
        print(feature_imp)

        a4_dims = (10, 30)
        fig, ax = plt.subplots(figsize=a4_dims)
        sns.set(font_scale=0.0001)
        sns.barplot(ax=ax, x=feature_imp, y=feature_imp.index, )
        plt.title("Feature Importance Score")
        plt.ylabel('Features')
        plt.title("Auditory Feature Importance")
        plt.legend()
        plt.savefig("feature_importance_graph", dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches='tight', pad_inches=0.1,
                    frameon=None, metadata=None)

        plt.show()
    return score_list


# calculate scores for consecutive runs of the random forest and average the result
scores = []
avg_score = 0
num_iterations = 1000
for n in range(num_iterations):
    scores = run_regressor(scores, 0)
print(scores)
for score in scores:
    avg_score += score
avg_score = avg_score/len(scores)
print("avg_score= " + str(avg_score))
