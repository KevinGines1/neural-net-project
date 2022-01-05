import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# * load Data from the files
dataset = pd.read_csv('../../student-data/student-mat.csv', sep=';')
# dataset = pd.read_csv('../../student-data/student-por.csv', sep=';')

column_names_index = dataset.columns
columnNames = [name for name in column_names_index]
# print(dataset)
feature_names = columnNames[:-1]
# print(feature_names)
label_name = columnNames[-1]
class_names = ['FAIL', 'PASS']

independentVariables = dataset.iloc[:,:-1].values
dependentVariables = dataset.iloc[:,-1].values

# * encode binary vars
LE1 = LabelEncoder()
independentVariables[:,0] = np.array(LE1.fit_transform(independentVariables[:,0]))
independentVariables[:,1] = np.array(LE1.fit_transform(independentVariables[:,1]))
independentVariables[:,3] = np.array(LE1.fit_transform(independentVariables[:,3]))
independentVariables[:,4] = np.array(LE1.fit_transform(independentVariables[:,4]))
independentVariables[:,5] = np.array(LE1.fit_transform(independentVariables[:,5]))
independentVariables[:,15] = np.array(LE1.fit_transform(independentVariables[:,15]))
independentVariables[:,16] = np.array(LE1.fit_transform(independentVariables[:,16]))
independentVariables[:,17] = np.array(LE1.fit_transform(independentVariables[:,17]))
independentVariables[:,18] = np.array(LE1.fit_transform(independentVariables[:,18]))
independentVariables[:,19] = np.array(LE1.fit_transform(independentVariables[:,19]))
independentVariables[:,20] = np.array(LE1.fit_transform(independentVariables[:,20]))
independentVariables[:,21] = np.array(LE1.fit_transform(independentVariables[:,21]))
independentVariables[:,22] = np.array(LE1.fit_transform(independentVariables[:,22]))

# * encode categorical vars
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[8, 9, 10, 11])],remainder="passthrough")
independentVariables = np.array(ct.fit_transform(independentVariables))

for i in range(len(dependentVariables)):
  if(dependentVariables[i] >= 10):
    dependentVariables[i] = 1
  else:
    dependentVariables[i] = 0

# * split the dataset into training and testing
indep_train,indep_test,dep_train,dep_test = train_test_split(independentVariables,dependentVariables,test_size=0.4,random_state=0) # 60% training 40% testing

# * standardize the values
sc = StandardScaler()
indep_train = sc.fit_transform(indep_train)
indep_test = sc.transform(indep_test)

# * initialize the ANN

ann = tf.keras.models.Sequential()

# * add hidden layers
ann.add(tf.keras.layers.Dense(input_dim = 45, units=31, activation='sigmoid', kernel_initializer='uniform'))

# add output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# compile the ANN
ann.compile(optimizer = 'adam', loss='binary_crossentropy', metrics='accuracy')

# fitting ANN
ann.fit(indep_train, dep_train, batch_size = 32, epochs = 1000)

# evaluate
ann.evaluate(indep_test, dep_test)