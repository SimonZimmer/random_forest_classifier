import pandas

dataframe = pandas.read_csv('/Users/simonzimmermann/dev/random_forest_regressor/datasets/Auswertung_HV.csv')

values = dataframe.iloc[5, 1:-1]
values.to_pickle("/Users/simonzimmermann/dev/random_forest_regressor/datasets/pkl/brunnensounds_mean.pkl")
