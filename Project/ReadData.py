import json
import pandas as pd
import os
import numpy as np
import glob
# Make all the json files for innings data into a dataframe
# inningsDataFrames = []
# pathToInnings = 'C:/Users/John Steward/Documents/GitHub/BachelorProject/Project/ipl_2022_innings_data/InningsData/*'
# inningsList = glob.glob(pathToInnings)
#
# for file in inningsList:
#     data = pd.read_json(file)
#     inningsDataFrames.append(data)
#
# allInnings = pd.concat(inningsDataFrames, ignore_index=True)
# allInnings.to_pickle('inningsData.pkl')
# Make all the ball data into a dataframe
ballDataFrames = []
pathToBall = 'C:/Users/John Steward/Documents/GitHub/BachelorProject/Project/ipl_2022_data/BallData/*'
ballList = glob.glob(pathToBall)
batsmanList = []
bowlerList = []
for file in ballList:
    # Load the json file into a dataframe and flatten the nested dictionaries
    with open(file, 'r') as f:
        data = json.loads(f.read())
    df = pd.json_normalize(data, max_level=3)
    # Create a list of players to correspond to a feature vector for each (so far will be a separate dataframe)
    batsmanName = df["match.battingTeam.batsman.name"]
    if batsmanName not in batsmanList:
        batsmanList.append(batsmanName)
    bowlerName = df["match.bowlingTeam.bowler.name"]
    if bowlerName not in bowlerList:
        bowlerList.append(bowlerName)

    # TODO: Create a list of dataframes, one for each player as a feature vector, corresponding to batsmanList and
    #       bowlerList
    #  Make all string data categorical (like shotPlayed, attacked...)

    # Drop consistent or unnecessary columns
    df.drop(['country', 'format', 'international', 'tourName', 'match.name'], axis=1, inplace=True)
    ballDataFrames.append(df)

allBalls = pd.concat(ballDataFrames, ignore_index=True)
allBalls.to_pickle('ballData.pkl')
allBalls.info()
df = pd.read_pickle('ballData.pkl')
df.info()

# pathToHawkeye = 'C:/Users/John Steward/Documents/GitHub/BachelorProject/Project/HawkeyeStats-main/mensIPLHawkeyeStats.csv'
# hawkeyeStats = pd.read_csv(pathToHawkeye)
# hawkeyeStats.to_pickle('hawkeyeStats.pkl')
# df = pd.read_pickle('hawkeyeStats.pkl')
# df.info()