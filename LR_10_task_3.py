
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import preprocessing

input_file = 'traffic_data.txt'
data = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line[:-1].split(',')
        data.append(items)

data = np.array(data)

label_encoder = []
X_encoded = np.empty(data.shape)

for i, item in enumerate(data[0]):
    if item.isdigit():
        X_encoded[:, i] = data[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(data[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
regressor = ExtraTreesRegressor(**params)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print("Mean absolute error:", round(mean_absolute_error(y_test, y_pred), 2))

test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
test_datapoint_encoded = [-1] * len(test_datapoint)

count = 0
for i, item in enumerate(test_datapoint):
    if item.isdigit():
        test_datapoint_encoded[i] = int(item)
    else:
        test_datapoint_encoded[i] = int(label_encoder[count].transform([item])[0])
        count += 1

test_datapoint_encoded = np.array(test_datapoint_encoded)

print("Predicted traffic:", int(regressor.predict([test_datapoint_encoded])[0]))

# Спочатку перевіримо доступні міста в даних
print("\nДоступні міста в даних:")
for i, encoder in enumerate(label_encoder):
    if i == 2:  # Індекс для міста
        print("Міста:", encoder.classes_)
        break

test_datapoints = [
    ['Monday', '12:30', 'Atlanta', 'yes'],
    ['Friday', '18:15', 'Chicago', 'no'],
    ['Sunday', '14:00', 'Los Angeles', 'yes']
]

print("\nДодаткові прогнози:")
for test_point in test_datapoints:
    test_encoded = [-1] * len(test_point)
    count = 0
    for i, item in enumerate(test_point):
        if item.isdigit():
            test_encoded[i] = int(item)
        else:
            test_encoded[i] = int(label_encoder[count].transform([item])[0])
            count += 1
    
    test_encoded = np.array(test_encoded)
    prediction = int(regressor.predict([test_encoded])[0])
    print(f"{test_point} -> Прогнозований трафік: {prediction}")
