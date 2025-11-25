
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

housing_data = fetch_california_housing()
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

regressor = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4),
    n_estimators=400, 
    random_state=7
)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print("ADABOOST REGRESSOR")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

feature_importances = regressor.feature_importances_
feature_names = housing_data.feature_names

feature_importances = 100.0 * (feature_importances / max(feature_importances))

index_sorted = np.flipud(np.argsort(feature_importances))

pos = np.arange(index_sorted.shape[0]) + 0.5

plt.figure(figsize=(12, 8))
plt.bar(pos, feature_importances[index_sorted], align='center')
plt.xticks(pos, [feature_names[i] for i in index_sorted], rotation=45)
plt.ylabel('Relative Importance')
plt.title('Важливість ознак за допомогою AdaBoost Regressor')
plt.tight_layout()
plt.savefig('feature_importance_10_2.png')
plt.show()

print("\nАНАЛІЗ ВАЖЛИВОСТІ ОЗНАК:")
print("=" * 50)
for i in index_sorted:
    print(f"{feature_names[i]}: {feature_importances[i]:.2f}%")

print("\nВИСНОВКИ:")
print("Ознаки з найбільшою важливістю:")
for i in range(3):
    print(f"{i+1}. {feature_names[index_sorted[i]]} - {feature_importances[index_sorted[i]]:.2f}%")

print("\nОзнаки з найменшою важливістю:")
for i in range(-3, 0):
    print(f"{feature_names[index_sorted[i]]} - {feature_importances[index_sorted[i]]:.2f}%")
