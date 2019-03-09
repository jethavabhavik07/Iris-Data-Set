# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
df = pd.read_csv("C:\iris.csv")
df.dtypes

df.describe()
df['Petal width'].plot.hist()
plt.show()
sns.pairplot(df,hue='Class')

features = df[['Sepal length','Sepal width','Petal length','Petal width']].values
classes = df['Class'].values

(train_feat,test_feat,train_classes,test_classes)=train_test_split(features,classes,train_size = 0.7,random_state = 1)

dec = DecisionTreeClassifier()
dec.fit(train_feat,train_classes)

pred = dec.predict(test_feat)
print('Accuracy : ',metrics.accuracy_score(test_classes,pred))

#Predicting a single input feature

sepl = input("Sepal Length:")
sepw = input("Sepal width:")
petl = input("Petal Length:")
petw = input("Petal width:")

pr = dec.predict(np.column_stack([sepl,sepw,petl,petw]))
print('Predicted Species is : ',pr)


import os
os.environ["path"] = "c:/Program files(x86)/Graphvz2.38/bin"

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
feat_col = ['Sepal length','Sepal width','Petal length','Petal width']
export_graphviz(dec,out_file = dot_data,filled = True , rounded = True, special_characters = True,feature_name = feat_col,class_name=['Setosa','Versicolor','Virginica'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalues())
graph.write_png('iris.png')
Image(graph.create_png())





















