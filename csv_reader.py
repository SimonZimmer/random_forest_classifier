
import pandas
import pickle as pkl

# brunnensounds
dataframe = pandas.read_csv('/Users/simonzimmermann/dev/random_forest_regressor/datasets/brunnensounds/csv/Auswertung_HV.csv')

values = dataframe.iloc[5, 1:-1]
values.to_pickle("/Users/simonzimmermann/dev/random_forest_regressor/datasets/brunnensounds/pkl/brunnensounds_mean.pkl")

# vaillant dataset 1
dataframe = pandas.read_csv('/Users/simonzimmermann/dev/random_forest_regressor/datasets/vaillant_1/csv/20190115_Zusammenfassung_V8_SV_NDA.csv')
values = dataframe.iloc[1:35,8]
values.to_pickle("/Users/simonzimmermann/dev/random_forest_regressor/datasets/vaillant_1/pkl/vaillant_1.pkl")

target = pkl.load(open("/Users/simonzimmermann/dev/random_forest_regressor/datasets/vaillant_1/pkl/vaillant_1.pkl"))

