import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
from sklearn.model_selection import train_test_split

ballData = pd.read_pickle("ballData.pkl")
ballData.info()


hist = ballData.plot.hist(column=['match.delivery.scoringInformation.score'], bins=7, range=[0, 7])
plt.show()
# Split data into train(80%) and test(20%)
# train, test = train_test_split(ballData, test_size=0.2, shuffle=True)
# print(train.shape, test.shape)
# # extract the feature that we want to predict
# target = train['match.delivery.scoringInformation.score']
# model = gpc()
# model.fit(train, target)