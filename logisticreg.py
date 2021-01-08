from sklearn.datasets import load_iris
from sklearn import tree
#first import the library
import pandas as pd
# datasert load
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
dataset = pd.read_csv(url, names=names)

#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
# dataset = pd.read_csv(data, names=names)
print(dataset.head()) #it prints 20 rows of data

# label encoding the data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['species']= le.fit_transform(dataset['species'])
#slicing
X_features_input = dataset.iloc[:, :-1].values #features[rows, columms]
print(X_features_input)
y_label_output = dataset.iloc[:, 4].values #labels
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_features_input, y_label_output, test_size=0.20, random_state=5)
#x_train = 80% of our features data(input)
#x_test = 20% of our features data(input)
#y_train = 80% of our lable data(output)
#y_test = 20 % of pur lable data(output)
#imported the algorithms from library

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=150)
#you can change these itration to check max. accuracy
logreg.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
predicted= logreg.predict(X_test)
print(predicted)
print(y_test)
print('accuracy: ' , accuracy_score(y_test,predicted))