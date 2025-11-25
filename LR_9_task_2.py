
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

input_file = 'data_imbalance.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])

plt.figure(figsize=(10, 8))
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black', edgecolors='black', linewidth=1, marker='o')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='o')
plt.title('Вхідні дані')
plt.savefig('input_data_9_2.png')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
classifier_no_balance = ExtraTreesClassifier(**params)
classifier_no_balance.fit(X_train, y_train)

visualize_classifier(classifier_no_balance, X_train, y_train, 'Training dataset - Without Balance')
visualize_classifier(classifier_no_balance, X_test, y_test, 'Test dataset - Without Balance')

class_names = ['Class-0', 'Class-1']
print("БЕЗ балансування класів")
print("#" * 40)
print("Classifier performance on training dataset")
print(classification_report(y_train, classifier_no_balance.predict(X_train), target_names=class_names))
print("#" * 40)
print("Classifier performance on test dataset")
print(classification_report(y_test, classifier_no_balance.predict(X_test), target_names=class_names))
print("#" * 40)

params_balanced = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0, 'class_weight': 'balanced'}
classifier_balanced = ExtraTreesClassifier(**params_balanced)
classifier_balanced.fit(X_train, y_train)

visualize_classifier(classifier_balanced, X_train, y_train, 'Training dataset - With Balance')
visualize_classifier(classifier_balanced, X_test, y_test, 'Test dataset - With Balance')

print("З балансуванням класів")
print("#" * 40)
print("Classifier performance on training dataset")
print(classification_report(y_train, classifier_balanced.predict(X_train), target_names=class_names))
print("#" * 40)
print("Classifier performance on test dataset")
print(classification_report(y_test, classifier_balanced.predict(X_test), target_names=class_names))
print("#" * 40)

print("ПОРІВНЯННЯ РЕЗУЛЬТАТІВ")
print("=" * 50)

y_pred_no_balance = classifier_no_balance.predict(X_test)
y_pred_balanced = classifier_balanced.predict(X_test)

report_no_balance = classification_report(y_test, y_pred_no_balance, target_names=class_names, output_dict=True)
report_balanced = classification_report(y_test, y_pred_balanced, target_names=class_names, output_dict=True)

print("Без балансування - Accuracy:", report_no_balance['accuracy'])
print("З балансуванням - Accuracy:", report_balanced['accuracy'])
print()
print("Без балансування - Class-0 F1-score:", report_no_balance['Class-0']['f1-score'])
print("З балансуванням - Class-0 F1-score:", report_balanced['Class-0']['f1-score'])
print()
print("Без балансування - Class-1 F1-score:", report_no_balance['Class-1']['f1-score'])
print("З балансуванням - Class-1 F1-score:", report_balanced['Class-1']['f1-score'])
