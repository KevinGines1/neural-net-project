import os
import matplotlib.pyplot as plt
import tensorflow as tf
# from data_utils import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# print("TensorFlow version: {}".format(tf.__version__))
# print("Eager execution: {}".format(tf.executing_eagerly()))

# * load Data from the files
# mathDataHeaders, mathDataList = loadData('../../student-data/student-mat.csv')
mathDataset = pd.read_csv('../../student-data/student-mat.csv', sep=';')

column_names_index = mathDataset.columns
columnNames = [name for name in column_names_index]
print(mathDataset)
feature_names = columnNames[:-1]
print(feature_names)
label_name = columnNames[-1]
class_names = ['FAIL', 'PASS']

independentVariables = mathDataset.iloc[:,:-1].values
# print(independentVariables[0:5])
dependentVariables = mathDataset.iloc[:,-1].values
# print(dependentVariables)

# * encode binary vars
LE1 = LabelEncoder()
independentVariables[:,0] = np.array(LE1.fit_transform(independentVariables[:,0]))
independentVariables[:,1] = np.array(LE1.fit_transform(independentVariables[:,1]))
independentVariables[:,3] = np.array(LE1.fit_transform(independentVariables[:,3]))
independentVariables[:,4] = np.array(LE1.fit_transform(independentVariables[:,4]))
independentVariables[:,5] = np.array(LE1.fit_transform(independentVariables[:,5]))
independentVariables[:,8] = np.array(LE1.fit_transform(independentVariables[:,8]))
independentVariables[:,9] = np.array(LE1.fit_transform(independentVariables[:,9]))
independentVariables[:,10] = np.array(LE1.fit_transform(independentVariables[:,10]))
independentVariables[:,11] = np.array(LE1.fit_transform(independentVariables[:,11]))
independentVariables[:,15] = np.array(LE1.fit_transform(independentVariables[:,15]))
independentVariables[:,16] = np.array(LE1.fit_transform(independentVariables[:,16]))
independentVariables[:,17] = np.array(LE1.fit_transform(independentVariables[:,17]))
independentVariables[:,18] = np.array(LE1.fit_transform(independentVariables[:,18]))
independentVariables[:,19] = np.array(LE1.fit_transform(independentVariables[:,19]))
independentVariables[:,20] = np.array(LE1.fit_transform(independentVariables[:,20]))
independentVariables[:,21] = np.array(LE1.fit_transform(independentVariables[:,21]))
independentVariables[:,22] = np.array(LE1.fit_transform(independentVariables[:,22]))
# print(independentVariables[:8])

# * encode categorical vars
# ct =ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[8, 9, 10, 11])],remainder="passthrough")
# independentVariables = np.array(ct.fit_transform(independentVariables))
# print(independentVariables[:8])


# * split the dataset into training and testing
indep_train,indep_test,dep_train,dep_test = train_test_split(independentVariables,dependentVariables,test_size=0.4,random_state=0) # 60% training 40% testing

# * standardize the values
sc = StandardScaler()
indep_train = sc.fit_transform(indep_train)
indep_test = sc.transform(indep_test)

# * initialize the ANN

ann = tf.keras.models.Sequential()

# add first hidden latyer
# ann.add(tf.keras.layers.Dense(input_dim = 45, units=31, activation='relu', kernel_initializer='uniform'))
ann.add(tf.keras.layers.Dense(input_dim = 32, units=22, activation='relu', kernel_initializer='uniform'))

# add output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# compile the ANN
ann.compile(optimizer = 'adam', loss='binary_crossentropy', metrics='accuracy')

# fitting ANN
ann.fit(indep_train, dep_train, batch_size = 32, epochs = 100)

# predicting
# print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1,50000]])) > 0.5)