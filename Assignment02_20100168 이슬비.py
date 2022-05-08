#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:01:32 2022

@author: drizzle0171
"""

# DO NOT CHANGE THIS PART
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
%matplotlib inline

data=pd.read_csv('https://drive.google.com/uc?export=download&id=1ZdJM0WQTaw3D2JRSNgV1KSyN2_-b0o2k', index_col=0)
trn, test=train_test_split(data,test_size=0.2, shuffle=True, random_state=11)

###############################################################################
# 1
###############################################################################
train_X = trn.drop(['y_target'], axis=1)
train_y = trn.y_target
test_X = test.drop('y_target', axis=1)
test_y = test.y_target

multiNB = MultinomialNB()
multiNB.fit(train_X, train_y)
y_pred = multiNB.predict(test_X)

#tp, tn, fn, fp 분류
tp, tn, fn, fp = 0, 0, 0, 0
for target, pred in zip(test_y, y_pred):
    if (target==1 and pred==1):
        tp +=1
    elif (target==1 and pred==0):
        fn +=1
    elif (target==0 and pred==0):
        tn +=1
    elif (target==0 and pred==1):
        fp +=1

#accuracy: (tp+tn)/(tp+fn+fp+tn)
accuracy = (tp+tn)/(tp+tn+fn+fp)
print(accuracy)
print(multiNB.score(test_X, test_y))

#recall: tp/(tp+fn)
recall = tp/(tp+fn)
print(recall)

#precision: tp/(tp+fp)
precision = tp/(tp+fp)
print(precision)

#f1: 2*precision x recall / (precision+recall)
f1 = 2*precision*recall/(precision+recall)
print(f1)

###############################################################################

space = multiNB.feature_log_prob_[0].argsort()[:10]
computer = multiNB.feature_log_prob_[1].argsort()[:10]
column = list(data)

for i, j, rank in zip(space, computer, range(1, 11)):
    print(f'space {rank}: {column[i]}')
    print(f'computer {rank}: {column[j]}')

###############################################################################

words = data.drop('y_target', axis=1).sum(axis=0).sort_values(ascending=False)[:30]
top30 = list(words.index)
column_test = list(test_X)

#space
space30 = []
for i in range(30):
    index = column_test.index(top30[i])
    space30.append(multiNB.feature_log_prob_[0][index])
    
#computer
computer30 = []
for i in range(30):
    index = column.index(top30[i])
    computer30.append(multiNB.feature_log_prob_[1][index])
    
fig, ax = plt.subplots()
y = range(30)
ax.barh(y, space30, color='blue', alpha=0.5, label='Space')
ax.barh(y, computer30, color='red', alpha=0.5, label='Computer')
ax.set_yticks(y)
ax.set_yticklabels(top30)
ax.invert_xaxis()
ax.set_xlabel('Probaility')
ax.set_title('Top30')
plt.legend()

plt.show()

###############################################################################

test_binary = test_X.values.tolist()
train_binary = train_X.values.tolist()

for i in range(len(train_binary)):
    for j in range(len(train_binary[i])):
        if train_binary[i][j] > 0:
            train_binary[i][j] = 1
        else:
            train_binary[i][j] = 0
            
for i in range(len(test_binary)):
    for j in range(len(test_binary[i])):
        if test_binary[i][j] > 0:
            test_binary[i][j] = 1
        else:
            test_binary[i][j] = 0
            
dfTest_binary = pd.DataFrame(test_binary, columns=column_test)
dfTrain_binary = pd.DataFrame(train_binary, columns=column_test)

bernoulliNB = BernoulliNB()
bernoulliNB.fit(dfTrain_binary, train_y)
y_pred_bin = bernoulliNB.predict(dfTest_binary)

#tp, tn, fn, fp 분류
tp, tn, fn, fp = 0, 0, 0, 0
for target, pred in zip(test_y, y_pred_bin):
    if (target==1 and pred==1):
        tp +=1
    elif (target==1 and pred==0):
        fn +=1
    elif (target==0 and pred==0):
        tn +=1
    elif (target==0 and pred==1):
        fp +=1

#accuracy: (tp+tn)/(tp+fn+fp+tn)
accuracy = (tp+tn)/(tp+tn+fn+fp)
print(accuracy)
print(bernoulliNB.score(dfTest_binary, test_y))

#recall: tp/(tp+fn)
recall = tp/(tp+fn)
print(recall)

#precision: tp/(tp+fp)
precision = tp/(tp+fp)
print(precision)

#f1: 2*precision x recall / (precision+recall)
f1 = 2*precision*recall/(precision+recall)
print(f1)

###############################################################################

cutoff = list(np.arange(0.1, 1, 0.05))
for i in range(len(cutoff)):
    cutoff[i] = round(cutoff[i], 2)

probability = multiNB.predict_proba(test_X)

test_y = test_y.tolist()
over = []
accuracy = []
cnt = 0
for i in range(len(cutoff)):
    for j in range(len(probability)):
        if cutoff[i] < probability[j][1]:
            over.append(1)
        else:
            over.append(0)
    for k in range(len(over)):
        if over[k] == test_y[k]:
            cnt += 1
    accu = cnt/len(test_y)*100
    accuracy.append(accu)
    over=[]
    accu=0
    cnt=0

fig=plt.figure(figsize=(15,5))
plt.scatter(cutoff,accuracy)
plt.xlabel('cutoff')
plt.ylabel('accuracy')
plt.xticks(cutoff)
plt.show()

###############################################################################
# 2
###############################################################################

# DO NOT CHANGE THIS PART
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split


data=pd.read_csv('https://drive.google.com/uc?export=download&id=1ds25l6300RnG7MMzRmtrHVdlUXmVsUvi')

train, test=train_test_split(X, y, test_size=0.2, shuffle=True)

train_X = train.drop(['target'], axis=1)
train_y = train.target
test_X = test.drop('target', axis=1)
test_y = test.target

DecisionTree = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_split=50, min_samples_leaf=25)
DecisionTree.fit(train_X, train_y)
y_pred = DecisionTree.predict(test_X)
y_prob = DecisionTree.predict_proba(test_X)

#tp, tn, fn, fp 분류: class 1
tp, tn, fn, fp = 0, 0, 0, 0
for target, pred in zip(test_y, y_pred):
    if (target==1 and pred==1):
        tp +=1
    elif (target==1 and pred==0):
        fn +=1
    elif (target==0 and pred==0):
        tn +=1
    elif (target==0 and pred==1):
        fp +=1

#accuracy: (tp+tn)/(tp+fn+fp+tn)
accuracy = (tp+tn)/(tp+tn+fn+fp)
print(accuracy)
print(DecisionTree.score(test_X, test_y))

#recall: class 1
recall = tp/(tp+fn)
print(recall)

#precision: class 1 
precision = tp/(tp+fp)
print(precision)

#tp, tn, fn, fp 분류: class 0
tp, tn, fn, fp = 0, 0, 0, 0
for target, pred in zip(test_y, y_pred):
    if (target==0 and pred==0):
        tp +=1
    elif (target==0 and pred==1):
        fn +=1
    elif (target==1 and pred==1):
        tn +=1
    elif (target==1 and pred==0):
        fp +=1
        
# Recall: class 0
recall = tp/(tp+fn)
print(recall)

# Precision: class 0
precision = tp/(tp+fp)
print(precision)

###############################################################################

feature_name = list(train_X.columns)
class_name = ['0', '1']

from sklearn import tree
import matplotlib.pyplot as plt

tree.plot_tree(DecisionTree, feature_names = feature_name, class_names=class_name, filled=True)

###############################################################################
# 3
###############################################################################

# DO NOT CHANGE THIS PART
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

%matplotlib inline

data=pd.read_csv('https://drive.google.com/uc?export=download&id=1Yo_VK8ntbGWe6Z8cEKNrWINUrpxqMc0Z')


trn,val=train_test_split(data, test_size=0.2,random_state=78)

train_X = trn.drop('area', axis=1)
train_y = trn.area
for i in range(len(train_y)):
    if train_y.values[i]>0:
        train_y.values[i] = 1
        
test_X = val.drop('area', axis=1)
test_y = val.area
for i in range(len(test_y)):
    if test_y.values[i]>0:
        test_y.values[i] = 1

days = [0]*7
for i in range(len(data)):
    if data.day.values[i] == 'mon':
        days[0] += 1
    elif data.day.values[i] == 'tue':
        days[1] += 1
    elif data.day.values[i] == 'wed':
        days[2] += 1
    elif data.day.values[i] == 'thu':
        days[3] += 1
    elif data.day.values[i] == 'fri':
        days[4] += 1
    elif data.day.values[i] == 'sat':
        days[5] += 1
    elif data.day.values[i] == 'sun':
        days[6] += 1

week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(len(days)):
    ratio = days[i]/sum(days)*100
    print(f'{week[i]} = {ratio}')

###############################################################################

# Month to Number: train
for i in range(len(train_X)):
    if train_X.month.values[i] == 'jan':
        train_X.month.values[i] = 1
    elif train_X.month.values[i] == 'feb':
        train_X.month.values[i] = 2
    elif train_X.month.values[i] == 'mar':
        train_X.month.values[i] = 3
    elif train_X.month.values[i] == 'apr':
        train_X.month.values[i] = 4
    elif train_X.month.values[i] == 'may':
        train_X.month.values[i] = 5
    elif train_X.month.values[i] == 'jun':
        train_X.month.values[i] = 6
    elif train_X.month.values[i] == 'jul':
        train_X.month.values[i] = 7
    elif train_X.month.values[i] == 'aug':
        train_X.month.values[i] = 8
    elif train_X.month.values[i] == 'sep':
        train_X.month.values[i] = 9
    elif train_X.month.values[i] == 'oct':
        train_X.month.values[i] = 10
    elif train_X.month.values[i] == 'nov':
        train_X.month.values[i] = 11
    elif train_X.month.values[i] == 'dec':
        train_X.month.values[i] = 12

# Month to Number: test
for i in range(len(test_X)):
    if test_X.month.values[i] == 'jan':
        test_X.month.values[i] = 1
    elif test_X.month.values[i] == 'feb':
        test_X.month.values[i] = 2
    elif test_X.month.values[i] == 'mar':
        test_X.month.values[i] = 3
    elif test_X.month.values[i] == 'apr':
        test_X.month.values[i] = 4
    elif test_X.month.values[i] == 'may':
        test_X.month.values[i] = 5
    elif test_X.month.values[i] == 'jun':
        test_X.month.values[i] = 6
    elif test_X.month.values[i] == 'jul':
        test_X.month.values[i] = 7
    elif test_X.month.values[i] == 'aug':
        test_X.month.values[i] = 8
    elif test_X.month.values[i] == 'sep':
        test_X.month.values[i] = 9
    elif test_X.month.values[i] == 'oct':
        test_X.month.values[i] = 10
    elif test_X.month.values[i] == 'nov':
        test_X.month.values[i] = 11
    elif test_X.month.values[i] == 'dec':
        test_X.month.values[i] = 12

# Day to Number: 1~7 train
for i in range(len(train_X)):
    if train_X.day.values[i] == 'mon':
        train_X.day.values[i] = 1
    elif train_X.day.values[i] == 'tue':
        train_X.day.values[i] = 2
    elif train_X.day.values[i] == 'wed':
        train_X.day.values[i] = 3
    elif train_X.day.values[i] == 'thu':
        train_X.day.values[i] = 4
    elif train_X.day.values[i] == 'fri':
        train_X.day.values[i] = 5
    elif train_X.day.values[i] == 'sat':
        train_X.day.values[i] = 6
    elif train_X.day.values[i] == 'sun':
        train_X.day.values[i] = 7

# Day to Number: 1~7 test
for i in range(len(test_X)):
    if test_X.day.values[i] == 'mon':
        test_X.day.values[i] = 1
    elif test_X.day.values[i] == 'tue':
        test_X.day.values[i] = 2
    elif test_X.day.values[i] == 'wed':
        test_X.day.values[i] = 3
    elif test_X.day.values[i] == 'thu':
        test_X.day.values[i] = 4
    elif test_X.day.values[i] == 'fri':
        test_X.day.values[i] = 5
    elif test_X.day.values[i] == 'sat':
        test_X.day.values[i] = 6
    elif test_X.day.values[i] == 'sun':
        test_X.day.values[i] = 7

neighbors = [1,5,9,11]
for i in range(len(neighbors)):
    knn = KNeighborsClassifier(n_neighbors=neighbors[i], metric = 'manhattan')
    knn.fit(train_X, train_y)
    score = knn.score(test_X, test_y)
    print(f'K = {neighbors[i]} \n:{score}')
    
###############################################################################

from sklearn.metrics import pairwise_distances

sam1=test_X[0:1].values
testX = test_X.values
print(sam1[:])
x1=[testX[i,0] for i in range(1,len(testX))]
x2=[testX[i,1] for i in range(1,len(testX))]
x3=[testX[i,2] for i in range(1,len(testX))]
x4=[testX[i,3] for i in range(1,len(testX))]
x5=[testX[i,4] for i in range(1,len(testX))]
x6=[testX[i,5] for i in range(1,len(testX))]
x7=[testX[i,6] for i in range(1,len(testX))]
x8=[testX[i,7] for i in range(1,len(testX))]
x9=[testX[i,8] for i in range(1,len(testX))]
x10=[testX[i,9] for i in range(1,len(testX))]

train=pd.DataFrame(data={'x1':x1,'x2':x2,'x3':x3,'x4':x4,'x5':x5,'x6':x6,'x7':x7,'x8':x8,'x9':x9,'x10':x10})

train['dist'] = pairwise_distances(sam1,train, metric='euclidean')[0]
sort_train=train.sort_values(by=['dist'],axis=0)

num=20

sample20_ind=sort_train.iloc[range(num)].index.values
knn_ind=sample20_ind[:10]
not_nn_ind=sample20_ind[10:]

plt.scatter(train.loc[not_nn_ind,'x8'],train.loc[not_nn_ind,'x9'],c='k')
plt.scatter(train.loc[knn_ind,'x9'],train.loc[knn_ind,'x9'],c='b')
plt.ylim((0, 50))
plt.show()

###############################################################################

def standard_regularization(list): #표준정규화함수
    mu=np.mean(list)
    var=np.var(list)
    for i in range(len(list)): #표준화
        list[i]=(list[i]-mu)/np.sqrt(var)
    return list

X = pd.concat([train_X, test_X])
X = X.values

Xcopy= X.copy()

x1=standard_regularization([X[i][0] for i in range(len(Xcopy))])
x2=standard_regularization([X[i][1] for i in range(len(Xcopy))])
x3=standard_regularization([X[i][2] for i in range(len(Xcopy))])
x4=standard_regularization([X[i][3] for i in range(len(Xcopy))])
x5=standard_regularization([X[i][4] for i in range(len(Xcopy))])
x6=standard_regularization([X[i][5] for i in range(len(Xcopy))])
x7=standard_regularization([X[i][6] for i in range(len(Xcopy))])
x8=standard_regularization([X[i][7] for i in range(len(Xcopy))])
x9=standard_regularization([X[i][8] for i in range(len(Xcopy))])
x10=standard_regularization([X[i][9] for i in range(len(Xcopy))])

newX=[[x1[i],x2[i],x3[i],x4[i],x5[i],x6[i],x7[i],x8[i],x9[i],x10[i]] for i in range(len(x1))]
newX=np.array(newX)
y = data['area']
for i in range(len(y)):
    if y.values[i]>0:
        y.values[i] = 1
y = y.values
        
trnX,valX,trnY,valY=train_test_split(newX,y,test_size=0.2,random_state=10, stratify=y)

knnmd=KNeighborsClassifier(n_neighbors=1)
knnmd.fit(trnX,trnY)
y_pred=knnmd.predict(valX)
print("k=1: ",knnmd.score(valX,valY))

knnmd=KNeighborsClassifier(n_neighbors=5)
knnmd.fit(trnX,trnY)
y_pred=knnmd.predict(valX)
print("k=5: ",knnmd.score(valX,valY))

knnmd=KNeighborsClassifier(n_neighbors=9)
knnmd.fit(trnX,trnY)
y_pred=knnmd.predict(valX)
print("k=9: ",knnmd.score(valX,valY))

knnmd=KNeighborsClassifier(n_neighbors=11)
knnmd.fit(trnX,trnY)
y_pred=knnmd.predict(valX)
print("k=11: ",knnmd.score(valX,valY))

###############################################################################

sam1=test_X[0:1].values
testX = test_X.values
print(sam1[:])
x1=[valX[i,0] for i in range(1,len(valX))]
x2=[valX[i,1] for i in range(1,len(valX))]
x3=[valX[i,2] for i in range(1,len(valX))]
x4=[valX[i,3] for i in range(1,len(valX))]
x5=[valX[i,4] for i in range(1,len(valX))]
x6=[valX[i,5] for i in range(1,len(valX))]
x7=[valX[i,6] for i in range(1,len(valX))]
x8=[valX[i,7] for i in range(1,len(valX))]
x9=[valX[i,8] for i in range(1,len(valX))]
x10=[valX[i,9] for i in range(1,len(valX))]

train=pd.DataFrame(data={'x1':x1,'x2':x2,'x3':x3,'x4':x4,'x5':x5,'x6':x6,'x7':x7,'x8':x8,'x9':x9,'x10':x10})

train['dist']=pairwise_distances(sam1,train,metric='euclidean')[0]
sort_train=train.sort_values(by=['dist'],axis=0)

num=20

sample20_ind=sort_train.iloc[range(num)].index.values
knn_ind=sample20_ind[:10]
not_nn_ind=sample20_ind[10:]

plt.scatter(train.loc[not_nn_ind,'x8'],train.loc[not_nn_ind,'x9'],c='k')
plt.scatter(train.loc[knn_ind,'x8'],train.loc[knn_ind,'x9'],c='b')
plt.xlim((-2,2))
plt.ylim((-2,2))
plt.show()

