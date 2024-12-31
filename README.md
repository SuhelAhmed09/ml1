import numpy as nm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

data_set= pd.read_csv('User_Data.csv')

x= data_set.iloc[:, [2,3]].values
y= data_set.iloc[:, 4].values
print(x)
print(y)

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)

st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

y_pred= classifier.predict(x_test)
print(y_pred)

cm= confusion_matrix(y_test, y_pred)
print(cm)

accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
tree.plot_tree(clf)

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(x, y)
tree.plot_tree(clf)
plt.figure(figsize=(45,60))
tree.plot_tree(clf,filled=True)
