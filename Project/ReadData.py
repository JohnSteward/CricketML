import json
import pandas as pd
import os
import numpy as np
import glob
# Make all the json files for innings data into a dataframe
inningsDataFrames = []
pathToInnings = 'C:/Users/John Steward/Documents/GitHub/BachelorProject/Project/ipl_2022_innings_data/InningsData/*'
inningsList = glob.glob(pathToInnings)

for file in inningsList:
    data = pd.read_json(file)
    inningsDataFrames.append(data)

allInnings = pd.concat(inningsDataFrames, ignore_index=True)
allInnings.to_pickle('inningsData.pkl')
# # Make all the ball data into a dataframe
ballDataFrames = []
pathToBall = 'C:/Users/John Steward/Documents/GitHub/BachelorProject/Project/ipl_2022_data/BallData/*'
ballList = glob.glob(pathToBall)

for file in ballList:
    data = pd.read_json(file)
    ballDataFrames.append(data)

allBalls = pd.concat(ballDataFrames, ignore_index=True)
allBalls.to_pickle('ballData.pkl')
allBalls.info()
df = pd.read_pickle('ballData.pkl')
df.info()

pathToHawkeye = 'C:/Users/John Steward/Documents/GitHub/BachelorProject/Project/HawkeyeStats-main/mensIPLHawkeyeStats.csv'
hawkeyeStats = pd.read_csv(pathToHawkeye)
hawkeyeStats.to_pickle('hawkeyeStats.pkl')
df = pd.read_pickle('hawkeyeStats.pkl')
df.info()