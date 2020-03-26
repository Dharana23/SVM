from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
Y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

svc = SVC(kernel='rbf', gamma=1.0)
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
cm_rbf01 = confusion_matrix(y_test, y_predict)

svc = SVC(kernel='rbf', gamma=10.0)
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
cm_rbf10 = confusion_matrix(y_test, y_predict)

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
cm_linear = confusion_matrix(y_test, y_predict)

svc = SVC(kernel='poly')
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
cm_poly = confusion_matrix(y_test, y_predict)

svc = SVC(kernel='sigmoid')
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
cm_sigmoid = confusion_matrix(y_test, y_predict)
