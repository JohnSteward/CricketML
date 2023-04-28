import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF

ballData = pd.read_pickle("ballData.pkl")
ballData.info()
# hist = ballData.plot.hist(column=['match.delivery.scoringInformation.score'], bins=8, range=[-1, 7])
# plt.show()

fileName = "model50%1.pkl"

# Split data into train(70%) and test(30%)
train, test = train_test_split(ballData, test_size=0.5, shuffle=True)
testy, exclusion = train_test_split(test, test_size=0.6, shuffle=True)

#realTest, exclusion = train_test_split(test, test_size=0.8, shuffle=True)
#print(train.shape, realTest.shape)
#train.columns = train.columns.str.replace(' ', '')
#train['match.delivery.scoringInformation.score'] = train['match.delivery.scoringInformation.score'].astype('int32')
# extract the feature that we want to predict
train.info()
target = train['match.delivery.scoringInformation.score'].values
testTarget = testy['match.delivery.scoringInformation.score'].values
#print(target.dtype)
kernel = 1.0*RBF(1.0)
model = gpc(kernel=kernel, n_jobs=-1).fit(train, target)
print("trained")

pickle.dump(model, open(fileName, "wb"))
print(model.score(testy, testTarget))
