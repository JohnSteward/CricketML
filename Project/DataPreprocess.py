import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

ballData = pd.read_pickle("ballData.pkl")
ballData.info()
# pca = PCA(n_components=50)
# pca.fit(ballData)

