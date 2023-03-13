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
batsmanList = []
bowlerList = []
pathToBall = 'C:/Users/John Steward/Documents/GitHub/BachelorProject/Project/ipl_2022_data/BallData/*'
ballList = glob.glob(pathToBall)

for file in ballList:
    # Load the json file into a dataframe and flatten the nested dictionaries
    with open(file, 'r') as f:
        data = json.loads(f.read())
    df = pd.json_normalize(data)
    # Drop consistent or unnecessary columns
    df.drop(['country', 'format', 'international', 'tourName', 'match.name'], axis=1, inplace=True)
    ballDataFrames.append(df)
 # Create a list of players to correspond to a feature vector for each (so far will be a separate dataframe)
    batsmanName = df["match.battingTeam.batsman.name"]
    if batsmanName.item() not in batsmanList:
        batsmanList.append(batsmanName.item())
    bowlerName = df["match.bowlingTeam.bowler.name"]
    if bowlerName.item() not in bowlerList:
        bowlerList.append(bowlerName.item())
    # TODO: Create a list of dataframes, one for each player as a feature vector, corresponding to batsmanList and
    #       bowlerList
    #  Make all string data categorical (like shotPlayed, attacked...)

# need categorical data for this and delivery type
attackedNo = 0
defendNo = 0
for batsman in batsmanList:
    totalRuns = 0
    totalBalls = 0
    for bowler in bowlerList:
        for file in ballDataFrames:
            if (file["match.battingTeam.batsman.name"].item() == batsman) and (file["match.bowlingTeam.bowler.name"].item() == bowler):
                totalRuns += file["match.delivery.scoringInformation.score"].item()
                totalBalls += 1
            elif file["match.battingTeam.batsman.name"].item() == batsman:
                totalRuns += file["match.delivery.scoringInformation.score"].item()
                totalBalls += 1

for bowler in bowlerList:
    totalBalls = 0
    totalRuns = 0
    totalDots = 0
    totalOnes = 0
    totalTwos = 0
    totalThrees = 0
    totalFours = 0
    totalSixes = 0
    totalWicketsOrDropped = 0
    for file in ballDataFrames:
        if file["match.bowlingTeam.bowler.name"].item() == bowler:
            totalBalls += 1
            if file["match.delivery.scoringInformation.score"].item() == 1:
                totalOnes += 1
                totalRuns += 1
            elif file["match.delivery.scoringInformation.score"].item() == 2:
                totalTwos += 1
                totalRuns += 2
            elif file["match.delivery.scoringInformation.score"].item() == 3:
                totalThrees += 1
                totalRuns += 3
            elif file["match.delivery.scoringInformation.score"].item() == 4:
                totalFours += 1
                totalRuns += 4
            elif file["match.delivery.scoringInformation.score"].item() == 6:
                totalSixes += 1
                totalRuns += 6
            if (file["match.delivery.scoringInformation.wicket.isWicket"].item() == True) or (file["match.delivery.additionalEventInformation.dropped"].item() == True):
                totalWicketsOrDropped += 1


# allBalls = pd.concat(ballDataFrames, ignore_index=True)
# allBalls.to_pickle('ballData.pkl')
# allBalls.info()
# df = pd.read_pickle('ballData.pkl')
# df.info()

# pathToHawkeye = 'C:/Users/John Steward/Documents/GitHub/BachelorProject/Project/HawkeyeStats-main/mensIPLHawkeyeStats.csv'
# hawkeyeStats = pd.read_csv(pathToHawkeye)
# hawkeyeStats.to_pickle('hawkeyeStats.pkl')
# df = pd.read_pickle('hawkeyeStats.pkl')
# df.info()