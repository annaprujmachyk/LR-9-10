
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

def visualize_classifier(classifier, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='black', linewidth=1, cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.show()

input_file = 'data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])
class_2 = np.array(X[y == 2])

plt.figure(figsize=(10, 8))
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='o')
plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='^')
plt.title('Вхідні дані')
plt.savefig('input_data_9_1.png')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
classifier_rf = RandomForestClassifier(**params)
classifier_rf.fit(X_train, y_train)

visualize_classifier(classifier_rf, X_train, y_train, 'Random Forest - Training dataset')
visualize_classifier(classifier_rf, X_test, y_test, 'Random Forest - Test dataset')

class_names = ['Class-0', 'Class-1', 'Class-2']
print("Random Forest - Training dataset")
print(classification_report(y_train, classifier_rf.predict(X_train), target_names=class_names))
print("Random Forest - Test dataset")
print(classification_report(y_test, classifier_rf.predict(X_test), target_names=class_names))

classifier_erf = ExtraTreesClassifier(**params)
classifier_erf.fit(X_train, y_train)

visualize_classifier(classifier_erf, X_train, y_train, 'Extra Trees - Training dataset')
visualize_classifier(classifier_erf, X_test, y_test, 'Extra Trees - Test dataset')

print("Extra Trees - Training dataset")
print(classification_report(y_train, classifier_erf.predict(X_train), target_names=class_names))
print("Extra Trees - Test dataset")
print(classification_report(y_test, classifier_erf.predict(X_test), target_names=class_names))

test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]])

print("Random Forest Confidence Measure:")
for datapoint in test_datapoints:
    probabilities = classifier_rf.predict_proba([datapoint])[0]
    predicted_class = 'Class-' + str(np.argmax(probabilities))
    print(f'Datapoint: {datapoint}, Predicted class: {predicted_class}, Probabilities: {probabilities}')

print("Extra Trees Confidence Measure:")
for datapoint in test_datapoints:
    probabilities = classifier_erf.predict_proba([datapoint])[0]
    predicted_class = 'Class-' + str(np.argmax(probabilities))
    print(f'Datapoint: {datapoint}, Predicted class: {predicted_class}, Probabilities: {probabilities}')

visualize_classifier(classifier_rf, test_datapoints, [0]*len(test_datapoints), 'Test datapoints - Random Forest')
visualize_classifier(classifier_erf, test_datapoints, [0]*len(test_datapoints), 'Test datapoints - Extra Trees')
