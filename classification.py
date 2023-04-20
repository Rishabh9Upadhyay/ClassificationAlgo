import pandas as pd
import matplotlib.pyplot as plt
fruits = pd.read_table('./fruit_data.txt')
print(fruits.head())

print(fruits.shape)
print(fruits['fruit_name'].unique())
print(fruits['fruit_subtype'].unique())
print(fruits['fruit_subtype'].unique())
print(fruits.groupby('fruit_name').size())
a=fruits.groupby('fruit_name').size()


import seaborn as sns
sns.countplot(x=fruits['fruit_name'],label = 'Count')
plt.show()

fruits.drop('fruit_label',axis=1).plot(kind='bar', subplots=True, layout=(2,2), sharex=(2,2), sharey=False, figsize=(9,8), title='Box plot for input variable')
plt.savefig('fruit_box')
plt.show()          #we plot box plot to detect outlayers in dataset

import pylab as p1
fruits.drop('fruit_label',axis=1).hist(bins=30, figsize=(9,8))
p1.suptitle('Histogram for each numeric input variable')
plt.savefig('fruit_hist')
plt.show()

# now divide dataset into target and predictor variable
feature_names = ['mass', 'width', 'height', 'color_score']
x = fruits[feature_names]
y = fruits['fruit_label']

# Data splicing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)         #all the data will be in similar range
x_test = scaler.transform(x_test)


# Logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
print('Accuracy of logistic regression classifier on training set: {:.2f}'.format(logreg.score(x_train, y_train)))
print('Accuracy of Logistic regression classifier on testing set: {:.2f}'.format(logreg.score(x_test, y_test)))
# So it's depends on problem statement for which logistic regression is most suitable
# Accuracy of logistic regression classifier on training set: 0.75
# Accuracy of Logistic regression classifier on testing set: 0.47

# Decision tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(x_train, y_train)
print('Accuracy of Desion Tree classifier on traning set: {:.2f}'.format(clf.score(x_train, y_train)))
print('Accuracy of Desion Tree classifier on testing set: {:.2f}'.format(clf.score(x_test, y_test)))
# Accuracy of Desion Tree classifier on traning set: 1.00
# Accuracy of Desion Tree classifier on testing set: 0.67
# Decision trees are very good in training dataset 
# because of a process known as overfitting.
# But when it comes to clasifying the outcome ont the testing data set, the accuracy reduces



# KNN classifier
# from sklearn.neighbors import KNeighborsTransformer
# knn = KNeighborsTransformer(n_neighbors=5)
# knn.fit(x_train, y_train)
# y_pred = knn.predict(x_test)
# print(y_pred)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
print("Accuracy of KNN classifier on traning set: {:.2f}".format(knn.score(x_train, y_train)))
print("Accuracy of KNN classifier on tesing set: {:.2f}".format(knn.score(x_test, y_test)))
# Accuracy of KNN classifier on traning set: 0.95
# Accuracy of KNN classifier on tesing set: 1.00
# KNN classifier  classifiedf our dataset more accyratilly
# we'll look at the predictions that the KNN classifier mean





# NaiveBayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(x_train, y_train)))
print('Accuracy of GNB classifier on testing set: {:.2f}'.format(gnb.score(x_test, y_test)))
# Accuracy of GNB classifier on training set: 0.86
# Accuracy of GNB classifier on testing set: 0.67

# support vector machine classifier
from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(x_train, y_train)))
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(x_test, y_test)))
# Accuracy of SVM classifier on training set: 0.91
# Accuracy of SVM classifier on training set: 0.80


gamma = 'auto'
# confusion metrics for knn classifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = knn.predict(x_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# So confusion metrics is a table that is often 
# used to describe the performance of a classification modal
# It actully represents a table of actual and predicted value