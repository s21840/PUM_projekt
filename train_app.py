from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import classification_report,accuracy_score

import numpy as np

import pickle
import os 

import pandas as pd
base_data = pd.read_csv('card_transactions_400k.csv')
cols = ["fraud", "distance_from_home", "distance_from_last_transaction", "ratio_to_median_purchase_price", "repeat_retailer", "used_chip", "used_pin_number", "online_order"]
data = base_data[cols].copy()

y = data.iloc[:,0]
x = data.iloc[:,1:8]

#stratify
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=55, stratify=y)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

def model(X_train, y_train):
    tree = DecisionTreeClassifier(criterion = 'gini',
                               max_depth =  7, 
                               splitter =  'best')
    tree.fit(X_train,y_train)
    print("Decision Tree: {0}".format(tree.score(X_train,y_train)))
    return tree

tree = model(X_train,y_train)
tree.fit(X_train, y_train)
prediction = tree.predict(X_test)

print ("Accuracy : ",accuracy_score(y_test,prediction)*100, " %")
print(classification_report(y_test, prediction))

pickle.dump(tree, open("tree_model.sv", 'wb'))


y1_predict = tree.predict(X_test)
print(y1_predict)
print(accuracy_score(y_test, y1_predict))
