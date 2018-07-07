# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 09:42:25 2017

@author: Paul
"""
pip install statistics
conda update scikit-learn
# csvReadTest.py #
# import statistics

from nltk.probability import FreqDist
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from operator import itemgetter
import statistics
import nltk
import csv
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 用自己抓的資料練習隨機森林
# 設定工作路徑
pathProg = 'C:\\Users\\user\\Documents\\R Lab\\Concrete Compressive Strength'
os.getcwd()
os.chdir(pathProg)
# 讀取檔案
df_load=pd.read_csv(pathProg + '\\Concrete_Data.csv',sep=',')
# 重新命名檔案
Concrete=df_load.copy()
# 將資料格式轉成dataframe
Concrete = pd.DataFrame(data=Concrete)
Concrete=Concrete.rename(columns={'Cement (component 1)(kg in a m^3 mixture)':'Cement',\
                                  'Blast Furnace Slag (component 2)(kg in a m^3 mixture)':'Blast Furnace Slag',\
                                  'Fly Ash (component 3)(kg in a m^3 mixture)':'Fly Ash',\
                                  'Water  (component 4)(kg in a m^3 mixture)':'Water',\
                                  'Superplasticizer (component 5)(kg in a m^3 mixture)':'Superplasticizer',\
                                  'Coarse Aggregate  (component 6)(kg in a m^3 mixture)':'Coarse Aggregate',\
                                  'Fine Aggregate (component 7)(kg in a m^3 mixture)':'Fine Aggregate',\
                                  'Age (day)':'Age',\
                                  'Concrete compressive strength(MPa, megapascals)':'Concrete compressive strength'})
# 描述性統計
# 類別變數-次數分配
#fdist = FreqDist(bank['job'])
# 數值變數-平均值、標準差、中位數、百分位數
summary=Concrete['age'].describe()
statistics.mean(Concrete['age'])
statistics.mean(Concrete['balance'])

# 看資料變數的型態(type)
Concrete.dtypes

# 看資料變數的名稱
var_names=list(Concrete)

# 將資料格式轉成陣列(array)
data_array=np.array(Concrete)

# 切分測試訓練集，隨機選取75%作為測試集、25%作為測試集
Concrete['is_train'] = np.random.uniform(0, 1, len(Concrete)) <= .75

train, test = Concrete[Concrete['is_train']==True], Concrete[Concrete['is_train']==False]
Concrete.head()

#==============================================================================
# list(train)
# tt=list(train.columns.values)
#==============================================================================

train_x_yz = train.iloc[:,:8]
train_y_ss, _= pd.factorize(train['Concrete compressive strength(MPa, megapascals) '])

clf = RandomForestClassifier(n_jobs=2)

# 利用迴圈寫一個挑選最佳mtry的函數
for m in range(1,9,1):
    rfc = RandomForestClassifier(n_estimators=10,n_jobs=1,max_features=m)
    rfc.fit(train_x,train_y)
    preds=rfc.predict(train_x)
    acc=rfc.score(train_x,train_y)
    print"變數",m,"個",",","預測正確率",acc

# 利用內建的語法挑選最佳參數mtry
rf = RandomForestClassifier(n_estimators=10,oob_score=True)
# rf.fit(train_x,train_y)
parameters = {'max_features':range(1,9,1)}

clf = GridSearchCV(estimator=rf, param_grid=parameters, scoring='roc_auc',iid=False)
clf.fit(train_x,train_y)
clf.grid_scores_
clf.best_params_
clf.best_score_
clf.best_estimator_
rf = clf.best_estimator_
# 配適模型(Fit Model)
rf = RandomForestClassifier(n_estimators=500,n_jobs=1,oob_score=True,max_features=3)
rf.fit(train_x,train_y)

test_x = test.iloc[:,:8]
test_y, _= pd.factorize(test['Concrete compressive strength(MPa, megapascals) '])

preds = rf.predict(test_x)
preds_table = pd.crosstab(test['Concrete compressive strength(MPa, megapascals) '], preds, rownames=['actual'], colnames=['preds'])
print(preds_table)

# 物件中的計算正確率方法 
rf.score(test_x,test_y)

# 自己寫的計算正確率方法
preds_table_a=np.array(preds_table)

correct_rate=100*float((preds_table_a[0,0]+preds_table_a[1,1]))/sum(sum(preds_table_a))
print(correct_rate)

# 畫出變數重要性的圖
var_per=rf.feature_importances_

indices = np.argsort(var_per)

var_per_sort=var_per[indices]

# 取出變數名稱
features = bank.columns[:9]
features

plt.figure(1)
plt.axis([0,0.3,0,16])
plt.title('Feature Importances')
plt.barh(range(len(indices)), var_per[indices], color='r', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()

# 再畫ROC





