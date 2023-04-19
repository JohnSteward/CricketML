import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from GPy.models import OneVsAllSparseClassification
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF

ballData = pd.read_pickle("ballData.pkl")


# hist = ballData.plot.hist(column=['match.delivery.scoringInformation.score'], bins=8, range=[-1, 7])
# plt.show()

# Split data into train(80%) and test(20%)
train, test = train_test_split(ballData, test_size=0.25, shuffle=True)


#realTest, exclusion = train_test_split(test, test_size=0.8, shuffle=True)
#print(train.shape, realTest.shape)
#train.columns = train.columns.str.replace(' ', '')
#train['match.delivery.scoringInformation.score'] = train['match.delivery.scoringInformation.score'].astype('int32')
# extract the feature that we want to predict

target = train['match.delivery.scoringInformation.score'].values
testTarget = test['match.delivery.scoringInformation.score'].values
#print(target.dtype)
kernel = 1.0*RBF(1.0)
model = gpc(kernel=kernel, n_jobs=-1).fit(train, target)
print("trained")
print(model.score(test.values, testTarget))
