from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

#fit the model
svm = SVC(kernel='linear')
svm.fit(x_train, y_train)

#predict in train and (train/)test samples
y_pred = svm.predict(x_test)


#accuracy
accuracy = accuracy_score(y_true, y_pred)
#f1
f1 = f1_score(y_true, y_pred)
