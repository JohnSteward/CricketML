import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from GPy.models import OneVsAllSparseClassification
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
from sklearn.model_selection import train_test_split

ballData = pd.read_pickle("ballData.pkl")



# hist = ballData.plot.hist(column=['match.delivery.scoringInformation.score'], bins=8, range=[-1, 7])
# plt.show()

# Split data into train(80%) and test(20%)
train, test = train_test_split(ballData, test_size=0.2, shuffle=True)
print(train.shape, test.shape)
index = 0
# extract the feature that we want to predict
train.columns = train.columns.str.replace(' ', '')
train['match.delivery.scoringInformation.score'] = train['match.delivery.scoringInformation.score'].astype('int32')
train.info()
target = train['match.delivery.scoringInformation.score'].values[np.newaxis]
print(target.dtype)
model = OneVsAllSparseClassification(train, target.T)