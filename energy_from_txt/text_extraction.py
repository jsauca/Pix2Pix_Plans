import os
import csv
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Lasso

data_folder = '../dataset/'

paths, coolings, heatings = [], [], []
with open('paths.csv', 'r') as reader:
    for line in list(reader)[1:]:
        sample = line.split(',')[1:]
        paths.append('/'.join(sample[0].split('/')[2:]))
        coolings.append(int(sample[1]))
        heatings.append(int(sample[2][:-2]))

heatings = np.array(heatings)
coolings = np.array(coolings)

room_types = ['living_room', 'kitchen', 'bedroom',
              'bathroom', 'restroom', 'washing_room',
              'office', 'closet', 'balcony',
              'corridor', 'dining_room', 'laundry_room',
              'PS']


def extract_file(file):
    area = 0
    nb_spaces = 0
    door_length, wall_length = 0, 0
    max_number = 0
    with open(file, 'r') as reader:
        lines_wall = []
        lines_door = []
        for i, line in enumerate(reader):
            line_main = line.split()
            line = [int(float(x)) for x in line_main[:4]]
            for coord in line:
                if coord > max_number:
                    max_number = coord
            if 'door' in line_main:
                door_length += (abs(line[0] - line[2]) +
                                abs(line[1] - line[3]))
            if 'wall' in line_main:
                wall_length += (abs(line[0] - line[2]) +
                                abs(line[1] - line[3]))
            if line_main[4] in room_types:
                area += (abs(line[0] - line[2]) * abs(line[1] - line[3]))

    return max_number, door_length, wall_length, area

# with open('energy_main.csv', 'w') as f :
#     writer = csv.writer(f, delimiter=',')
#     writer.writerow(['max_number', 'door_length', 'wall_length', 'area', 'cooling', 'heating'])
#     for index, path in enumerate(paths) :
#         file = os.listdir(os.path.join(data_folder, path))[0]
#         max_number, door_length, wall_length, area = extract_file(os.path.join(data_folder, path, file))
#         writer.writerow([max_number, door_length, wall_length, area, coolings[index], heatings[index]])


def process(path_data, training=True, MinMax=None):

    x_columns = ['max_number', 'door_length', 'wall_length', 'area']
    y1_column = ['cooling']
    y2_column = ['heating']

    data = pd.read_csv(path_data)
    X = data[x_columns]
    Y1 = data[y1_column]
    Y2 = data[y2_column]
    if training == True:
        """ Splitting """
        X_train_div, X_test_div, y1_train, y1_test = train_test_split(
            X, Y1, random_state=5, test_size=0.2)
        X_train_div, X_test_div, y2_train, y2_test = train_test_split(
            X, Y2, random_state=5, test_size=0.2)

        """ Scaling """
        MinMax = MinMaxScaler(feature_range=(0, 1))
        X_train_div = MinMax.fit_transform(X_train_div)
        X_test_div = MinMax.transform(X_test_div)
        return X_train_div, X_test_div, y1_train.to_numpy(), y1_test.to_numpy(), y2_train.to_numpy(), y2_test.to_numpy(), MinMax

    """ Scaling with previous scaler - If test """
    X = MinMax.transform(X)
    return X


path_data = 'energy_main.csv'
X_train_div, X_test_div, y1_train, y1_test, y2_train, y2_test, minmax = process(
    path_data, training=True)

# Parameters for grid search
# param_grid = {'max_features': ['auto', 'log2'],
#               'max_depth': [10]}  # 10,15,20,30,50,60, also possible change n_estimators
# 10,15,20,30,50,60, also possible change n_estimators
# param_grid = {'max_iter': [100000]}
# model
# model = RandomForestRegressor(random_state=5, n_estimators=10, n_jobs=-1)
# model = Lasso(alpha=1.0)
params = {'n_estimators': 500, 'max_depth': 100, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
grid_search_rf = GradientBoostingRegressor(**params)
""" Cooling """
# grid_search_rf = MultiOutputRegressor(GridSearchCV(
#     model, param_grid, cv=5, return_train_score=True, verbose=2))
# grid_search_rf.fit(X_train_div, y1_train)
grid_search_rf.fit(X_train_div, y1_train)
print('The Train R2 score for cooling load is', r2_score(
    y1_train, grid_search_rf.predict(X_train_div)))
print('The Test R2 score for cooling load is', r2_score(
    y1_test, grid_search_rf.predict(X_test_div)))

""" Heating """
# grid_search_rf2 = MultiOutputRegressor(GridSearchCV(
#     model, param_grid, cv=5, return_train_score=True, verbose=2))
grid_search_rf2 = GradientBoostingRegressor(**params)
grid_search_rf2.fit(X_train_div, y2_train)

print('The Train R2 score for heating load is', r2_score(
    y2_train, grid_search_rf2.predict(X_train_div)))
print('The Test R2 score for heating load is', r2_score(
    y2_test, grid_search_rf2.predict(X_test_div)))
