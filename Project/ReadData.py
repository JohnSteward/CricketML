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
deliveryList = []
extrasList = []
weightList = []
attackedList = []
playedList = []
addShotTypeList = []
pathToBall = 'C:/Users/John Steward/Documents/GitHub/BachelorProject/Project/ipl_2022_data/BallData/*'
ballList = glob.glob(pathToBall)

for file in ballList:
    # Load the json file into a dataframe and flatten the nested dictionaries
    with open(file, 'r') as f:
        data = json.loads(f.read())
    df = pd.json_normalize(data)
    # Drop consistent or unnecessary columns
    df.drop(['country', 'format', 'international', 'tourName', 'match.name', 'match.battingTeam.id',
             'match.battingTeam.batsman.id', 'match.bowlingTeam.bowlerPartner.id',
             'match.bowlingTeam.bowlerPartner.isRightHanded', 'match.bowlingTeam.bowlerPartner.name',
             'match.delivery.timecode', 'match.delivery.shotInformation.batsmanWeight'], axis=1, inplace=True)
    ballDataFrames.append(df)
 # Create a list of players to correspond to a feature vector for each (so far will be a separate dataframe)
    batsmanName = df["match.battingTeam.batsman.name"]
    if batsmanName.item() not in batsmanList:
        batsmanList.append(batsmanName.item())
    bowlerName = df["match.bowlingTeam.bowler.name"]
    if bowlerName.item() not in bowlerList:
        bowlerList.append(bowlerName.item())
    deliveryType = df["match.delivery.deliveryType"]
    if deliveryType.item() not in deliveryList:
        deliveryList.append(deliveryType.item())
    extras = df["match.delivery.scoringInformation.extrasType"]
    if extras.item() not in extrasList:
        extrasList.append(extras.item())
    attacked = df["match.delivery.shotInformation.shotAttacked"]
    if attacked.item() not in attackedList:
        attackedList.append(attacked.item())
    played = df["match.delivery.shotInformation.shotPlayed"]
    if played.item() not in playedList:
        playedList.append(played.item())
    addShot = df["match.delivery.shotInformation.shotTypeAdditional"]
    if addShot.item() not in addShotTypeList:
        addShotTypeList.append(addShot.item())
    # TODO: Create a list of dataframes, one for each player as a feature vector, corresponding to batsmanList and
    #       bowlerList
    #  Make all string data categorical (like shotPlayed, attacked...). Compile a list of all the delivery types,
    #  then run through the dataframe, replacing the strings with the index in the list.
print(deliveryList)
print(extrasList)
print(attackedList)
print(playedList)
print(addShotTypeList)

# Replacing all relevant strings with a numerical ID
for file in ballDataFrames:
    for i in range(len(deliveryList) - 1):
        if deliveryList[i] == file["match.delivery.deliveryType"].item():
            file["match.delivery.deliveryType"] = file["match.delivery.deliveryType"].replace(deliveryList[i], i)
    for j in range(len(extrasList) - 1):
        if extrasList[j] == file["match.delivery.scoringInformation.extrasType"].item():
            file["match.delivery.scoringInformation.extrasType"] = file["match.delivery.scoringInformation.extrasType"].replace(extrasList[j], j)
    for k in range(len(attackedList) - 1):
        if attackedList[k] == file["match.delivery.shotInformation.shotPlayed"].item():
            file["match.delivery.shotInformation.shotPlayed"] = file["match.delivery.shotInformation.shotPlayed"].replace(attackedList[k], k)
    for l in range(len(playedList) - 1):
        if playedList[l] == file["match.delivery.shotInformation.shotPlayed"].item():
            file["match.delivery.shotInformation.shotPlayed"] = file["match.delivery.shotInformation.shotPlayed"].replace(playedList[l], l)
    for m in range(len(addShotTypeList) - 1):
        if addShotTypeList[m] == file["match.delivery.shotInformation.shotTypeAdditional"].item():
            file["match.delivery.shotInformation.shotTypeAdditional"] = file["match.delivery.shotInformation.shotTypeAdditional"].replace(addShotTypeList[m], m)

for batsman in batsmanList:
    totalRuns = 0
    totalBalls = 0
    attackedNo = 0
    defendNo = 0
    for file in ballDataFrames:
        if file["match.battingTeam.batsman.name"].item() == batsman:
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