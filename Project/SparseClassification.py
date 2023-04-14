import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from GPy.models import OneVsAllSparseClassification, OneVsAllClassification

from sklearn.model_selection import train_test_split

ballData = pd.read_pickle("ballData.pkl")



# hist = ballData.plot.hist(column=['match.delivery.scoringInformation.score'], bins=8, range=[-1, 7])
# plt.show()

# Split data into train(80%) and test(20%)
train, test = train_test_split(ballData, test_size=0.2, shuffle=True)
print(train.shape, test.shape)
index = 0
# extract the feature that we want to predict
train.info()
target = train['match.delivery.scoringInformation.score'].values[np.newaxis]
print(type(target.T[0][0]))
trainy = []
for datapoint in train.values:
    for i in range(56, 79):
        datapoint[i] = float(datapoint[i])
    trainy.append(datapoint)
realTrain = np.asarray(trainy)
print(type(realTrain[0][70]))
for i in realTrain:
    for j in range(56, 79):
        if not (isinstance(i[j], float)):
            print(j)
print(realTrain)
model = OneVsAllSparseClassification(realTrain, target.T)
print("trained")