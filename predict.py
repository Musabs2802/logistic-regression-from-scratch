from regression import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from utils import calculate_accuracy

breast_cancer_data = load_breast_cancer()
X, y = breast_cancer_data.data, breast_cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = LogisticRegression()
regressor.fit(X_train, y_train)
y_predicted = regressor.predict(X_test)

accuracy = calculate_accuracy(y_predicted, y_test)

print("Accuracy: ", accuracy)