from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# decision tree
# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

# k neighbors
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

# train
my_classifier.fit(X_train, y_train)

predicitions = my_classifier.predict( X_test )

# print predicitions

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predicitions)