from sklearn import svm
import pandas as pd
import numpy as np

C=1e+03 #1000
gamma=1e-05 #0.0001
clf = svm.SVC(C=C, gamma=gamma, kernel='rbf', probability=True)

train = pd.read_csv('/diabetes_train.csv') #this is our dataset, please change the path for your case
test = pd.read_csv('/diabetes_test.csv')