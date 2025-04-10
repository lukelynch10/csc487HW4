from sklearn import svm
import pandas as pd
import numpy as np

train = pd.read_csv('/diabetes_train.csv') 
test = pd.read_csv('/diabetes_test.csv')

for i, chunk in enumerate(pd.read_csv(train, chunksize=2)):
        chunk.to_csv(f'{'train_{i}'}_{i+1}.csv', index=False)