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

print(deliveryList)
print(extrasList)
print(attackedList)
print(playedList)
print(addShotTypeList)

# Replacing all relevant strings with a numerical ID
for file in ballDataFrames:
    for i in range(len(deliveryList)):
        if deliveryList[i] == file["match.delivery.deliveryType"].item():
            file["match.delivery.deliveryType"] = file["match.delivery.deliveryType"].replace(deliveryList[i], i)
    for j in range(len(extrasList)):
        if extrasList[j] == file["match.delivery.scoringInformation.extrasType"].item():
            file["match.delivery.scoringInformation.extrasType"] = file["match.delivery.scoringInformation.extrasType"].replace(extrasList[j], j)
    for k in range(len(attackedList)):
        if attackedList[k] == file["match.delivery.shotInformation.shotPlayed"].item():
            file["match.delivery.shotInformation.shotPlayed"] = file["match.delivery.shotInformation.shotPlayed"].replace(attackedList[k], k)
    for l in range(len(playedList)):
        if playedList[l] == file["match.delivery.shotInformation.shotPlayed"].item():
            file["match.delivery.shotInformation.shotPlayed"] = file["match.delivery.shotInformation.shotPlayed"].replace(playedList[l], l)
    for m in range(len(addShotTypeList)):
        if addShotTypeList[m] == file["match.delivery.shotInformation.shotTypeAdditional"].item():
            file["match.delivery.shotInformation.shotTypeAdditional"] = file["match.delivery.shotInformation.shotTypeAdditional"].replace(addShotTypeList[m], m)

batsmanDataList = []
for batsman in batsmanList:
    totalRuns = 0
    runsVsSpin = 0
    runsVsSeam = 0
    runsVsMed = 0
    earlyOverRuns = 0
    midOverRuns = 0
    lateOverRuns = 0
    totalBalls = 0
    ballsSpin = 0
    ballsSeam = 0
    ballsMed = 0
    attackedNo = 0
    defendNo = 0
    attackSpin = 0
    attackSeam = 0
    attackMed = 0
    defendSpin = 0
    defendSeam = 0
    defendMed = 0
    for file in ballDataFrames:
        if file["match.battingTeam.batsman.name"].item() == batsman:
            totalRuns += file["match.delivery.scoringInformation.score"].item()
            totalBalls += 1
            if file["match.delivery.deliveryType"].item() == 0:
                ballsSpin += 1
                runsVsSpin += file["match.delivery.scoringInformation.score"].item()
                if file["match.delivery.shotInformation.shotPlayed"].item() == 0:
                    attackSpin += 1
                elif file["match.delivery.shotInformation.shotPlayed"].item() == 1:
                    defendSpin += 1
            elif file["match.delivery.deliveryType"].item() == 1:
                ballsSeam += 1
                runsVsSeam += file["match.delivery.scoringInformation.score"].item()
                if file["match.delivery.shotInformation.shotPlayed"].item() == 0:
                    attackSeam += 1
                elif file["match.delivery.shotInformation.shotPlayed"].item() == 1:
                    defendSeam += 1
            elif file["match.delivery.deliveryType"].item() == 2:
                ballsMed += 1
                runsVsMed += file["match.delivery.scoringInformation.score"].item()
                if file["match.delivery.shotInformation.shotPlayed"].item() == 0:
                    attackMed += 1
                elif file["match.delivery.shotInformation.shotPlayed"].item() == 1:
                    defendMed += 1

            if file["match.delivery.deliveryNumber.over"].item() <= 5:
                earlyOverRuns += file["match.delivery.scoringInformation.score"].item()
            elif (file["match.delivery.deliveryNumber.over"].item() > 5) and file["match.delivery.deliveryNumber.over"].item() <= 15:
                midOverRuns += file["match.delivery.scoringInformation.score"].item()
            elif file["match.delivery.deliveryNumber.over"].item() > 15:
                lateOverRuns += file["match.delivery.scoringInformation.score"].item()

            attackedNo = attackSpin + attackSeam + attackMed
            defendNo = defendSpin + defendSeam + defendMed
    if totalBalls > 0:
        strikeRate = (totalRuns/totalBalls)*100
        aggression = attackedNo / totalBalls
        passiveness = defendNo / totalBalls
        earlyRat = earlyOverRuns/totalRuns
        midRat = midOverRuns/totalRuns
        lateRat = lateOverRuns/totalRuns
    else:
        strikeRate = 0
        aggression = 0
        passiveness = 0
        earlyRat = 0
        midRat = 0
        lateRat = 0
    if ballsSpin > 0:
        strikeSpin = (runsVsSpin/ballsSpin)*100
        aggSpin = attackSpin / ballsSpin
        pasSpin = defendSpin / ballsSpin
    else:
        strikeSpin = 0
        aggSpin = 0
        pasSpin = 0
    if ballsSeam > 0:
        strikeSeam = (runsVsSeam/ballsSeam)*100
        aggSeam = attackSeam / ballsSeam
        pasSeam = defendSeam / ballsSeam
    else:
        strikeSeam = 0
        aggSeam = 0
        pasSeam = 0
    if ballsMed > 0:
        strikeMed = (runsVsMed/ballsMed)*100
        aggMed = attackMed / ballsMed
        pasMed = defendMed / ballsMed
    else:
        strikeMed = 0
        aggMed = 0
        pasMed = 0

    batData = {'Name': batsman, 'strikeRate': strikeRate, 'strikeSpin': strikeSpin, 'strikeSeam': strikeSeam,
               'strikeMed': strikeMed, 'earlyRat': earlyRat, 'midRat': midRat, 'lateRat': lateRat,
               'aggression': aggression, 'aggSpin': aggSpin, 'aggSeam': aggSeam, 'aggMed': aggMed,
               'passiveness': passiveness, 'pasSpin': pasSpin, 'pasSeam': pasSeam, 'pasMed': pasMed}
    df = pd.DataFrame(batData)
    batsmanDataList.append(df)

for file in ballDataFrames:
    for batsman in batsmanDataList:
        if file["match.battingTeam.batsman.name"].item() == batsman["Name"].item():
            file["match.battingTeam.batsman.name"].replace(batsman["Name"].item(), batsman)



bowlDataList = []
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
                totalDots += 1
            elif file["match.delivery.scoringInformation.score"].item() == 1:
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
    noOvers = totalBalls//6
    economy = totalRuns/noOvers
    dotRat = totalDots/totalBalls
    oneRat = totalOnes/totalBalls
    twoRat = totalTwos / totalBalls
    threeRat = totalThrees / totalBalls
    fourRat = totalFours / totalBalls
    sixRat = totalSixes / totalBalls
    wicRat = totalWicketsOrDropped/totalBalls

    bowlData = {'Name': bowler, 'economy': economy, 'dotRat': dotRat, 'oneRat': oneRat, 'twoRat': twoRat,
                'threeRat': threeRat, 'fourRat': fourRat, 'sixRat': sixRat, 'wicRat': wicRat}
    df = pd.DataFrame(bowlData)
    bowlDataList.append(df)

for file in ballDataFrames:
    for bowler in bowlDataList:
        if file["match.bowlingTeam.bowler.name"].item() == bowler["Name"].item():
            file["match.bowlingTeam.bowler.name"].replace(bowler["Name"].item(), bowler)

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