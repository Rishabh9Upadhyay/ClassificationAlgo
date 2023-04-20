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
# plt.show()

fruits.drop('fruit_label',axis=1).plot(kind='bar', subplots=True, layout=(2,2), sharex=(2,2), sharey=False, figsize=(9,8), title='Box plot for input variable')
plt.savefig('fruit_box')
# plt.show()          #we plot box plot to detect outlayers in dataset

import pylab as p1
fruits.drop('fruit_label',axis=1).hist(bins=30, figsize=(9,8))
p1.suptitle('Histogram for each numeric input variable')
plt.savefig('fruit_hist')
# plt.show()

# now divide dataset into target and predictor variable
feature_names = ['mass', 'width', 'height', 'color_score']
x = fruits[feature_names]
y = fruits['fruit_label']

# Data splicing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)         #all the data will be in similar range
x_test = scaler.transform(x_test)


# Logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
dataCBLR = pd.DataFrame({'Actual' : y_test,'Predicted' : y_pred})
dataCBLR.to_csv('my_data1.csv',index=False)
# there is very less data so will use directally
dataset1 = pd.read_csv('./my_data1.csv')
print(dataset1)
dataset1.plot(kind='bar',figsize=(11,7),xlabel='X-Axis',ylabel='Y-Axis',title='Now comparision b/w actual and predicted value of Logistic regression')
# plt.show()
print('Accuracy of logistic regression classifier on training set: {:.2f}'.format(logreg.score(x_train, y_train)))
print('Accuracy of Logistic regression classifier on testing set: {:.2f}'.format(logreg.score(x_test, y_test)))
# So it's depends on problem statement for which logistic regression is most suitable
# Accuracy of logistic regression classifier on training set: 0.75
# Accuracy of Logistic regression classifier on testing set: 0.47



# Decision tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(x_train, y_train)
y_pred = clf.predict(x_test)
dataset2 = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
dataset2.to_csv('./my_data2.csv',index=False)
dataset2 = pd.read_csv('./my_data2.csv')
print(dataset2)
dataset2.plot(kind='bar',figsize=(11,7),xlabel='X-Axis',ylabel='Y-Axis',title='Now comparision b/w actual and predicted value Decision tree')
plt.show()
# print('Accuracy of Desion Tree classifier on traning set: {:.2f}'.format(clf.score(x_train, y_train)))
print('Accuracy of Desion Tree classifier on testing set: {:.2f}'.format(clf.score(x_test, y_test)))
# Accuracy of Desion Tree classifier on traning set: 1.00
# Accuracy of Desion Tree classifier on testing set: 0.67
# Decision trees are very good in training dataset 
# because of a process known as overfitting.
# But when it comes to clasifying the outcome ont the testing data set, the accuracy reduces



# KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
dataset3 = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(dataset3)
dataset3.to_csv('my_data3.csv',index=False)
dataset3 = pd.read_csv('./my_data3.csv')
print('New way')
print(dataset3)
dataset3.plot(kind='bar',figsize=(11,7),xlabel='X-Axis',ylabel='Y-Axis',title='Now comparision b/w actual and predicted value of KNN')
# plt.show()
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
y_pred = gnb.predict(x_test)
dataset4 = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(dataset4)
dataset4.to_csv('./my_data4.csv',index=False)
dataset4 = pd.read_csv('my_data4.csv')
dataset4.plot(kind='bar',figsize=(11,7),xlabel='X-Axis',ylabel='Y-Axis',title='Now comparision b/w actual and predicted value of naive bayes')
# plt.show()
print('Accuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(x_train, y_train)))
print('Accuracy of GNB classifier on testing set: {:.2f}'.format(gnb.score(x_test, y_test)))
# Accuracy of GNB classifier on training set: 0.86
# Accuracy of GNB classifier on testing set: 0.67

# support vector machine classifier
from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
dataset5 = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
dataset5.to_csv('my_data5.csv',index=False)
dataset5 = pd.read_csv('my_data5.csv')
print(dataset5)
dataset5.plot(kind='bar',figsize=(11,7),xlabel='X-Axis',ylabel='Y-Axis',title='Now comparision b/w actual and predicted value of support vector machine classifier')
plt.show()
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

# '{:.2f}': This is a string formatting code that specifies how the accuracy score should be printed. In this case, '{:.2f}' means to print the accuracy as a floating point number with two decimal places.
