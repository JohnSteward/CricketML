import pickle
import time
import pandas as pd
import json
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

fileName = "Models/model70%1.pkl"

# Split data into train(70%) and test(30%)
# train, test = train_test_split(ballData, test_size=0.5, shuffle=True)
# testy, exclusion = train_test_split(test, test_size=0.4, shuffle=True)
# train.to_pickle('Training Sets/train50%2.pkl')
# testy.to_pickle('Test Sets/test50%2.pkl')
#realTest, exclusion = train_test_split(test, test_size=0.8, shuffle=True)
#print(train.shape, realTest.shape)
#train.columns = train.columns.str.replace(' ', '')
#train['match.delivery.scoringInformation.score'] = train['match.delivery.scoringInformation.score'].astype('int32')
# extract the feature that we want to predict
train = pd.read_pickle("Training Sets/train70%1.pkl")
test = pd.read_pickle("Test Sets/test70%1.pkl")
target = train['match.delivery.scoringInformation.score'].values
testTarget = test['match.delivery.scoringInformation.score'].values
train.drop(['match.delivery.scoringInformation.score'], axis=1, inplace=True)
test.drop(['match.delivery.scoringInformation.score'], axis=1, inplace=True)
# train.info()
#print(target.dtype)
kernel = 1.0*RBF(1.0)
startTime = time.time()
# model = gpc(kernel=kernel, n_jobs=-1).fit(train, target)
endTime = time.time()
totalTime = endTime - startTime
print("trained")
model = pickle.load(open(fileName, "rb"))
# pickle.dump(model, open(fileName, "wb"))
print(model.score(test, testTarget))

# Do overs 2, 12 and 16 of DLS match https://www.iplt20.com/match/2023/857' (total runs were 4, 8, 10)
# Bowlers: ARSHDEEP SINGH, HARPREET BRAR, ARSHDEEP SINGH
#Batsmen: (Mandeep Singh, ANUKUL ROY), (Andre Russell, VENKATESH IYER), (VENKATESH IYER, SHARDUL THAKUR)
batsman1 = pd.read_pickle("Batsman Files/Mandeep Singh.pkl")
batsman2 = pd.read_pickle("Batsman Files/ANUKUL ROY.pkl")
batsman3 = pd.read_pickle("Batsman Files/Andre Russell.pkl")
batsman4 = pd.read_pickle("Batsman Files/VENKATESH IYER.pkl")
batsman5 = pd.read_pickle("Batsman Files/VENKATESH IYER.pkl")
batsman6 = pd.read_pickle("Batsman Files/SHARDUL THAKUR.pkl")

bowler1 = pd.read_pickle("Bowler Files/ARSHDEEP SINGH.pkl")
bowler2 = pd.read_pickle("Bowler Files/HARPREET BRAR.pkl")
bowler3 = pd.read_pickle("Bowler Files/ARSHDEEP SINGH.pkl")

# Do overs 2, 8 and 16 of https://www.iplt20.com/match/2023/894 (total runs were 10, 7, 13)
# Bowlers: Andre Russell, Sunil Narine, Andre Russell
# Batsmen: (Wriddhiman Saha, SHUBMAN GILL), (SHUBMAN GILL, HARDIK PANDYA), (David Miller, VIJAY SHANKAR)
batsman7 = pd.read_pickle("Batsman Files/Wriddhiman Saha.pkl")
batsman8 = pd.read_pickle("Batsman Files/SHUBMAN GILL.pkl")
batsman9 = pd.read_pickle("Batsman Files/SHUBMAN GILL.pkl")
batsman10 = pd.read_pickle("Batsman Files/HARDIK PANDYA.pkl")
batsman11 = pd.read_pickle("Batsman Files/David Miller.pkl")
batsman12 = pd.read_pickle("Batsman Files/VIJAY SHANKAR.pkl")

bowler4 = pd.read_pickle("Bowler Files/Andre Russell.pkl")
bowler5 = pd.read_pickle("Bowler Files/Sunil Narine.pkl")
bowler6 = pd.read_pickle("Bowler Files/Andre Russell.pkl")

simulation1 = {
    "match": {
        "battingTeam": {
            "batsman": {
                "isRightHanded": True,
            },
            "batsmanPartner": {
                "isRightHanded": False,
            },
            "home": False,
        },
        "bowlingTeam": {
            "bowler": {
                "isRightHanded": False,
                "spell": 0
            },
            "home": True,
        },
        "delivery": {
            "deliveryNumber": {
                "ball": 1,
                "day": 1,
                "innings": 2,
                "over": 2
            },
            "deliveryType": 1,
            "fielderPosition": {
                "1st Slip": False,
                "2nd Slip": False,
                "3rd Slip": False,
                "4th Slip": False,
                "5th Slip": False,
                "Backwards Point": False,
                "Cover": False,
                "Cow Corner": False,
                "Deep Backwards Square Leg": False,
                "Deep Cover": False,
                "Deep Extra Cover": False,
                "Deep Mid Wicket": False,
                "Deep Point": False,
                "Deep Square Leg": False,
                "Extra Cover": False,
                "Fine Leg": False,
                "Fly Slip": False,
                "Gully": False,
                "Leg Gully": False,
                "Leg Slip": False,
                "Long Leg": False,
                "Long-off": False,
                "Long-on": False,
                "Mid Wicket": False,
                "Mid-off": False,
                "Mid-on": False,
                "Point": False,
                "Short Extra Cover": False,
                "Short Fine Leg": False,
                "Short Leg": False,
                "Short Mid-wicket": False,
                "Silly Mid-off": False,
                "Silly Point": False,
                "Square Leg": False,
                "Third Man": False,
                "Wicket Keeper": False
            },
            "isPavilionEnd": True,
            "round": True,
            "scoringInformation": {
                "extrasScore": 0,
                "extrasType": 0,
                "wicket": {
                    "isWicket": False,
                }
            },
            "shotInformation": {
                "shotAttacked": 0,
                "shotPlayed": 0,
                "shotTypeAdditional": 0
            },
        }
    }
}
simulation2 = {
    "match": {
        "battingTeam": {
            "batsman": {
                "isRightHanded": True,
            },
            "batsmanPartner": {
                "isRightHanded": False,
            },
            "home": False,
        },
        "bowlingTeam": {
            "bowler": {
                "isRightHanded": False,
                "spell": 0
            },
            "home": True,
        },
        "delivery": {
            "deliveryNumber": {
                "ball": 1,
                "day": 1,
                "innings": 2,
                "over": 12
            },
            "deliveryType": 2,
            "fielderPosition": {
                "1st Slip": False,
                "2nd Slip": False,
                "3rd Slip": False,
                "4th Slip": False,
                "5th Slip": False,
                "Backwards Point": False,
                "Cover": False,
                "Cow Corner": False,
                "Deep Backwards Square Leg": False,
                "Deep Cover": False,
                "Deep Extra Cover": False,
                "Deep Mid Wicket": False,
                "Deep Point": False,
                "Deep Square Leg": False,
                "Extra Cover": False,
                "Fine Leg": False,
                "Fly Slip": False,
                "Gully": False,
                "Leg Gully": False,
                "Leg Slip": False,
                "Long Leg": False,
                "Long-off": False,
                "Long-on": False,
                "Mid Wicket": False,
                "Mid-off": False,
                "Mid-on": False,
                "Point": False,
                "Short Extra Cover": False,
                "Short Fine Leg": False,
                "Short Leg": False,
                "Short Mid-wicket": False,
                "Silly Mid-off": False,
                "Silly Point": False,
                "Square Leg": False,
                "Third Man": False,
                "Wicket Keeper": False
            },
            "isPavilionEnd": True,
            "round": True,
            "scoringInformation": {
                "extrasScore": 0,
                "extrasType": 0,
                "wicket": {
                    "isWicket": False,
                }
            },
            "shotInformation": {
                "shotAttacked": 0,
                "shotPlayed": 0,
                "shotTypeAdditional": 0
            },
        }
    }
}

simulation3 = {
    "match": {
        "battingTeam": {
            "batsman": {
                "isRightHanded": False,
            },
            "batsmanPartner": {
                "isRightHanded": True,
            },
            "home": False,
        },
        "bowlingTeam": {
            "bowler": {
                "isRightHanded": False,
                "spell": 0
            },
            "home": True,
        },
        "delivery": {
            "deliveryNumber": {
                "ball": 1,
                "day": 1,
                "innings": 2,
                "over": 16
            },
            "deliveryType": 1,
            "fielderPosition": {
                "1st Slip": False,
                "2nd Slip": False,
                "3rd Slip": False,
                "4th Slip": False,
                "5th Slip": False,
                "Backwards Point": False,
                "Cover": False,
                "Cow Corner": False,
                "Deep Backwards Square Leg": False,
                "Deep Cover": False,
                "Deep Extra Cover": False,
                "Deep Mid Wicket": False,
                "Deep Point": False,
                "Deep Square Leg": False,
                "Extra Cover": False,
                "Fine Leg": False,
                "Fly Slip": False,
                "Gully": False,
                "Leg Gully": False,
                "Leg Slip": False,
                "Long Leg": False,
                "Long-off": False,
                "Long-on": False,
                "Mid Wicket": False,
                "Mid-off": False,
                "Mid-on": False,
                "Point": False,
                "Short Extra Cover": False,
                "Short Fine Leg": False,
                "Short Leg": False,
                "Short Mid-wicket": False,
                "Silly Mid-off": False,
                "Silly Point": False,
                "Square Leg": False,
                "Third Man": False,
                "Wicket Keeper": False
            },
            "isPavilionEnd": True,
            "round": True,
            "scoringInformation": {
                "extrasScore": 0,
                "extrasType": 0,
                "wicket": {
                    "isWicket": False,
                }
            },
            "shotInformation": {
                "shotAttacked": 0,
                "shotPlayed": 0,
                "shotTypeAdditional": 0
            },
        }
    }
}

simulation4 = {
    "match": {
        "battingTeam": {
            "batsman": {
                "isRightHanded": True,
            },
            "batsmanPartner": {
                "isRightHanded": True,
            },
            "home": False,
        },
        "bowlingTeam": {
            "bowler": {
                "isRightHanded": True,
                "spell": 0
            },
            "home": True,
        },
        "delivery": {
            "deliveryNumber": {
                "ball": 1,
                "day": 1,
                "innings": 2,
                "over": 2
            },
            "deliveryType": 1,
            "fielderPosition": {
                "1st Slip": False,
                "2nd Slip": False,
                "3rd Slip": False,
                "4th Slip": False,
                "5th Slip": False,
                "Backwards Point": False,
                "Cover": False,
                "Cow Corner": False,
                "Deep Backwards Square Leg": False,
                "Deep Cover": False,
                "Deep Extra Cover": False,
                "Deep Mid Wicket": False,
                "Deep Point": False,
                "Deep Square Leg": False,
                "Extra Cover": False,
                "Fine Leg": False,
                "Fly Slip": False,
                "Gully": False,
                "Leg Gully": False,
                "Leg Slip": False,
                "Long Leg": False,
                "Long-off": False,
                "Long-on": False,
                "Mid Wicket": False,
                "Mid-off": False,
                "Mid-on": False,
                "Point": False,
                "Short Extra Cover": False,
                "Short Fine Leg": False,
                "Short Leg": False,
                "Short Mid-wicket": False,
                "Silly Mid-off": False,
                "Silly Point": False,
                "Square Leg": False,
                "Third Man": False,
                "Wicket Keeper": False
            },
            "isPavilionEnd": True,
            "round": True,
            "scoringInformation": {
                "extrasScore": 0,
                "extrasType": 0,
                "wicket": {
                    "isWicket": False,
                }
            },
            "shotInformation": {
                "shotAttacked": 0,
                "shotPlayed": 0,
                "shotTypeAdditional": 0
            },
        }
    }
}
simulation5 = {
    "match": {
        "battingTeam": {
            "batsman": {
                "isRightHanded": True,
            },
            "batsmanPartner": {
                "isRightHanded": True,
            },
            "home": False,
        },
        "bowlingTeam": {
            "bowler": {
                "isRightHanded": True,
                "spell": 0
            },
            "home": True,
        },
        "delivery": {
            "deliveryNumber": {
                "ball": 1,
                "day": 1,
                "innings": 2,
                "over": 8
            },
            "deliveryType": 0,
            "fielderPosition": {
                "1st Slip": False,
                "2nd Slip": False,
                "3rd Slip": False,
                "4th Slip": False,
                "5th Slip": False,
                "Backwards Point": False,
                "Cover": False,
                "Cow Corner": False,
                "Deep Backwards Square Leg": False,
                "Deep Cover": False,
                "Deep Extra Cover": False,
                "Deep Mid Wicket": False,
                "Deep Point": False,
                "Deep Square Leg": False,
                "Extra Cover": False,
                "Fine Leg": False,
                "Fly Slip": False,
                "Gully": False,
                "Leg Gully": False,
                "Leg Slip": False,
                "Long Leg": False,
                "Long-off": False,
                "Long-on": False,
                "Mid Wicket": False,
                "Mid-off": False,
                "Mid-on": False,
                "Point": False,
                "Short Extra Cover": False,
                "Short Fine Leg": False,
                "Short Leg": False,
                "Short Mid-wicket": False,
                "Silly Mid-off": False,
                "Silly Point": False,
                "Square Leg": False,
                "Third Man": False,
                "Wicket Keeper": False
            },
            "isPavilionEnd": True,
            "round": True,
            "scoringInformation": {
                "extrasScore": 0,
                "extrasType": 0,
                "wicket": {
                    "isWicket": False,
                }
            },
            "shotInformation": {
                "shotAttacked": 0,
                "shotPlayed": 0,
                "shotTypeAdditional": 0
            },
        }
    }
}
simulation6 = {
    "match": {
        "battingTeam": {
            "batsman": {
                "isRightHanded": False,
            },
            "batsmanPartner": {
                "isRightHanded": True,
            },
            "home": False,
        },
        "bowlingTeam": {
            "bowler": {
                "isRightHanded": True,
                "spell": 0
            },
            "home": True,
        },
        "delivery": {
            "deliveryNumber": {
                "ball": 1,
                "day": 1,
                "innings": 2,
                "over": 16
            },
            "deliveryType": 1,
            "fielderPosition": {
                "1st Slip": False,
                "2nd Slip": False,
                "3rd Slip": False,
                "4th Slip": False,
                "5th Slip": False,
                "Backwards Point": False,
                "Cover": False,
                "Cow Corner": False,
                "Deep Backwards Square Leg": False,
                "Deep Cover": False,
                "Deep Extra Cover": False,
                "Deep Mid Wicket": False,
                "Deep Point": False,
                "Deep Square Leg": False,
                "Extra Cover": False,
                "Fine Leg": False,
                "Fly Slip": False,
                "Gully": False,
                "Leg Gully": False,
                "Leg Slip": False,
                "Long Leg": False,
                "Long-off": False,
                "Long-on": False,
                "Mid Wicket": False,
                "Mid-off": False,
                "Mid-on": False,
                "Point": False,
                "Short Extra Cover": False,
                "Short Fine Leg": False,
                "Short Leg": False,
                "Short Mid-wicket": False,
                "Silly Mid-off": False,
                "Silly Point": False,
                "Square Leg": False,
                "Third Man": False,
                "Wicket Keeper": False
            },
            "isPavilionEnd": True,
            "round": True,
            "scoringInformation": {
                "extrasScore": 0,
                "extrasType": 0,
                "wicket": {
                    "isWicket": False,
                }
            },
            "shotInformation": {
                "shotAttacked": 0,
                "shotPlayed": 0,
                "shotTypeAdditional": 0
            },
        }
    }
}

#Load the new datapoints into json formats

simulation1 = json.dumps(simulation1)
simulation2 = json.dumps(simulation2)
simulation3 = json.dumps(simulation3)
simulation4 = json.dumps(simulation4)
simulation5 = json.dumps(simulation5)
simulation6 = json.dumps(simulation6)

#Load the json files into dataframes
s1 = json.loads(simulation1)
sim1 = pd.json_normalize(s1)
s2 = json.loads(simulation2)
sim2 = pd.json_normalize(s2)
s3 = json.loads(simulation3)
sim3 = pd.json_normalize(s3)
s4 = json.loads(simulation4)
sim4 = pd.json_normalize(s4)
s5 = json.loads(simulation5)
sim5 = pd.json_normalize(s5)
s6 = json.loads(simulation6)
sim6 = pd.json_normalize(s6)

#Add all the batsman and bowler data to the simulation datapoints

sim1['strikeRate'] = batsman1['strikeRate'].item()
sim1['strikeSpin'] = batsman1['strikeSpin'].item()
sim1['strikeSeam'] = batsman1['strikeSeam'].item()
sim1['strikeMed'] = batsman1['strikeMed'].item()
sim1['earlyRat'] = batsman1['earlyRat'].item()
sim1['midRat'] = batsman1['midRat'].item()
sim1['lateRat'] = batsman1['lateRat'].item()
sim1['aggression'] = batsman1['aggression'].item()
sim1['aggSpin'] = batsman1['aggSpin'].item()
sim1['aggSeam'] = batsman1['aggSeam'].item()
sim1['aggMed'] = batsman1['aggMed'].item()
sim1['passiveness'] = batsman1['passiveness'].item()
sim1['pasSpin'] = batsman1['pasSpin'].item()
sim1['pasSeam'] = batsman1['pasSeam'].item()
sim1['pasMed'] = batsman1['pasMed'].item()
sim1['economy'] = bowler1['economy']
sim1['dotRat'] = bowler1['dotRat']
sim1['oneRat'] = bowler1['oneRat']
sim1['twoRat'] = bowler1['twoRat']
sim1['threeRat'] = bowler1['threeRat']
sim1['fourRat'] = bowler1['fourRat']
sim1['sixRat'] = bowler1['sixRat']
sim1['wicRat'] = bowler1['wicRat']

sim2['strikeRate'] = batsman3['strikeRate'].item()
sim2['strikeSpin'] = batsman3['strikeSpin'].item()
sim2['strikeSeam'] = batsman3['strikeSeam'].item()
sim2['strikeMed'] = batsman3['strikeMed'].item()
sim2['earlyRat'] = batsman3['earlyRat'].item()
sim2['midRat'] = batsman3['midRat'].item()
sim2['lateRat'] = batsman3['lateRat'].item()
sim2['aggression'] = batsman3['aggression'].item()
sim2['aggSpin'] = batsman3['aggSpin'].item()
sim2['aggSeam'] = batsman3['aggSeam'].item()
sim2['aggMed'] = batsman3['aggMed'].item()
sim2['passiveness'] = batsman3['passiveness'].item()
sim2['pasSpin'] = batsman3['pasSpin'].item()
sim2['pasSeam'] = batsman3['pasSeam'].item()
sim2['pasMed'] = batsman3['pasMed'].item()
sim2['economy'] = bowler2['economy']
sim2['dotRat'] = bowler2['dotRat']
sim2['oneRat'] = bowler2['oneRat']
sim2['twoRat'] = bowler2['twoRat']
sim2['threeRat'] = bowler2['threeRat']
sim2['fourRat'] = bowler2['fourRat']
sim2['sixRat'] = bowler2['sixRat']
sim2['wicRat'] = bowler2['wicRat']

sim3['strikeRate'] = batsman5['strikeRate'].item()
sim3['strikeSpin'] = batsman5['strikeSpin'].item()
sim3['strikeSeam'] = batsman5['strikeSeam'].item()
sim3['strikeMed'] = batsman5['strikeMed'].item()
sim3['earlyRat'] = batsman5['earlyRat'].item()
sim3['midRat'] = batsman5['midRat'].item()
sim3['lateRat'] = batsman5['lateRat'].item()
sim3['aggression'] = batsman5['aggression'].item()
sim3['aggSpin'] = batsman5['aggSpin'].item()
sim3['aggSeam'] = batsman5['aggSeam'].item()
sim3['aggMed'] = batsman5['aggMed'].item()
sim3['passiveness'] = batsman5['passiveness'].item()
sim3['pasSpin'] = batsman5['pasSpin'].item()
sim3['pasSeam'] = batsman5['pasSeam'].item()
sim3['pasMed'] = batsman5['pasMed'].item()
sim3['economy'] = bowler3['economy']
sim3['dotRat'] = bowler3['dotRat']
sim3['oneRat'] = bowler3['oneRat']
sim3['twoRat'] = bowler3['twoRat']
sim3['threeRat'] = bowler3['threeRat']
sim3['fourRat'] = bowler3['fourRat']
sim3['sixRat'] = bowler3['sixRat']
sim3['wicRat'] = bowler3['wicRat']

sim4['strikeRate'] = batsman7['strikeRate'].item()
sim4['strikeSpin'] = batsman7['strikeSpin'].item()
sim4['strikeSeam'] = batsman7['strikeSeam'].item()
sim4['strikeMed'] = batsman7['strikeMed'].item()
sim4['earlyRat'] = batsman7['earlyRat'].item()
sim4['midRat'] = batsman7['midRat'].item()
sim4['lateRat'] = batsman7['lateRat'].item()
sim4['aggression'] = batsman7['aggression'].item()
sim4['aggSpin'] = batsman7['aggSpin'].item()
sim4['aggSeam'] = batsman7['aggSeam'].item()
sim4['aggMed'] = batsman7['aggMed'].item()
sim4['passiveness'] = batsman7['passiveness'].item()
sim4['pasSpin'] = batsman7['pasSpin'].item()
sim4['pasSeam'] = batsman7['pasSeam'].item()
sim4['pasMed'] = batsman7['pasMed'].item()
sim4['economy'] = bowler4['economy']
sim4['dotRat'] = bowler4['dotRat']
sim4['oneRat'] = bowler4['oneRat']
sim4['twoRat'] = bowler4['twoRat']
sim4['threeRat'] = bowler4['threeRat']
sim4['fourRat'] = bowler4['fourRat']
sim4['sixRat'] = bowler4['sixRat']
sim4['wicRat'] = bowler4['wicRat']

sim5['strikeRate'] = batsman9['strikeRate'].item()
sim5['strikeSpin'] = batsman9['strikeSpin'].item()
sim5['strikeSeam'] = batsman9['strikeSeam'].item()
sim5['strikeMed'] = batsman9['strikeMed'].item()
sim5['earlyRat'] = batsman9['earlyRat'].item()
sim5['midRat'] = batsman9['midRat'].item()
sim5['lateRat'] = batsman9['lateRat'].item()
sim5['aggression'] = batsman9['aggression'].item()
sim5['aggSpin'] = batsman9['aggSpin'].item()
sim5['aggSeam'] = batsman9['aggSeam'].item()
sim5['aggMed'] = batsman9['aggMed'].item()
sim5['passiveness'] = batsman9['passiveness'].item()
sim5['pasSpin'] = batsman9['pasSpin'].item()
sim5['pasSeam'] = batsman9['pasSeam'].item()
sim5['pasMed'] = batsman9['pasMed'].item()
sim5['economy'] = bowler5['economy']
sim5['dotRat'] = bowler5['dotRat']
sim5['oneRat'] = bowler5['oneRat']
sim5['twoRat'] = bowler5['twoRat']
sim5['threeRat'] = bowler5['threeRat']
sim5['fourRat'] = bowler5['fourRat']
sim5['sixRat'] = bowler5['sixRat']
sim5['wicRat'] = bowler5['wicRat']

sim6['strikeRate'] = batsman11['strikeRate'].item()
sim6['strikeSpin'] = batsman11['strikeSpin'].item()
sim6['strikeSeam'] = batsman11['strikeSeam'].item()
sim6['strikeMed'] = batsman11['strikeMed'].item()
sim6['earlyRat'] = batsman11['earlyRat'].item()
sim6['midRat'] = batsman11['midRat'].item()
sim6['lateRat'] = batsman11['lateRat'].item()
sim6['aggression'] = batsman11['aggression'].item()
sim6['aggSpin'] = batsman11['aggSpin'].item()
sim6['aggSeam'] = batsman11['aggSeam'].item()
sim6['aggMed'] = batsman11['aggMed'].item()
sim6['passiveness'] = batsman11['passiveness'].item()
sim6['pasSpin'] = batsman11['pasSpin'].item()
sim6['pasSeam'] = batsman11['pasSeam'].item()
sim6['pasMed'] = batsman11['pasMed'].item()
sim6['economy'] = bowler6['economy']
sim6['dotRat'] = bowler6['dotRat']
sim6['oneRat'] = bowler6['oneRat']
sim6['twoRat'] = bowler6['twoRat']
sim6['threeRat'] = bowler6['threeRat']
sim6['fourRat'] = bowler6['fourRat']
sim6['sixRat'] = bowler6['sixRat']
sim6['wicRat'] = bowler6['wicRat']

# Run simulation 1
totalRuns = 0
first = False
for i in range(6):
    runs = model.predict(sim1)
    probs = model.predict_proba(sim1)
    sim1['match.delivery.deliveryNumber.ball'] = sim1['match.delivery.deliveryNumber.ball'].item() + 1
    print(int(runs))
    totalRuns += int(runs)
    if runs % 2 != 0:
        first = not (first)
    if first:
        sim1['strikeRate'] = batsman2['strikeRate'].item()
        sim1['strikeSpin'] = batsman2['strikeSpin'].item()
        sim1['strikeSeam'] = batsman2['strikeSeam'].item()
        sim1['strikeMed'] = batsman2['strikeMed'].item()
        sim1['earlyRat'] = batsman2['earlyRat'].item()
        sim1['midRat'] = batsman2['midRat'].item()
        sim1['lateRat'] = batsman2['lateRat'].item()
        sim1['aggression'] = batsman2['aggression'].item()
        sim1['aggSpin'] = batsman2['aggSpin'].item()
        sim1['aggSeam'] = batsman2['aggSeam'].item()
        sim1['aggMed'] = batsman2['aggMed'].item()
        sim1['passiveness'] = batsman2['passiveness'].item()
        sim1['pasSpin'] = batsman2['pasSpin'].item()
        sim1['pasSeam'] = batsman2['pasSeam'].item()
        sim1['pasMed'] = batsman2['pasMed'].item()
        temp = sim1['match.battingTeam.batsman.isRightHanded'].item()
        sim1['match.battingTeam.batsman.isRightHanded'] = sim1['match.battingTeam.batsmanPartner.isRightHanded'].item()
        sim1['match.battingTeam.batsmanPartner.isRightHanded'] = temp
    else:
        sim1['strikeRate'] = batsman1['strikeRate'].item()
        sim1['strikeSpin'] = batsman1['strikeSpin'].item()
        sim1['strikeSeam'] = batsman1['strikeSeam'].item()
        sim1['strikeMed'] = batsman1['strikeMed'].item()
        sim1['earlyRat'] = batsman1['earlyRat'].item()
        sim1['midRat'] = batsman1['midRat'].item()
        sim1['lateRat'] = batsman1['lateRat'].item()
        sim1['aggression'] = batsman1['aggression'].item()
        sim1['aggSpin'] = batsman1['aggSpin'].item()
        sim1['aggSeam'] = batsman1['aggSeam'].item()
        sim1['aggMed'] = batsman1['aggMed'].item()
        sim1['passiveness'] = batsman1['passiveness'].item()
        sim1['pasSpin'] = batsman1['pasSpin'].item()
        sim1['pasSeam'] = batsman1['pasSeam'].item()
        sim1['pasMed'] = batsman1['pasMed'].item()
print("Sim 1 yielded", totalRuns, "runs")

#Run simulation 2
totalRuns = 0
first = False
for i in range(6):
    runs = model.predict(sim2)
    probs = model.predict_proba(sim2)
    sim2['match.delivery.deliveryNumber.ball'] = sim2['match.delivery.deliveryNumber.ball'].item() + 1
    print(int(runs))
    print(probs)
    totalRuns += int(runs)
    if runs % 2 != 0:
        first = not (first)
    if first:
        sim2['strikeRate'] = batsman4['strikeRate'].item()
        sim2['strikeSpin'] = batsman4['strikeSpin'].item()
        sim2['strikeSeam'] = batsman4['strikeSeam'].item()
        sim2['strikeMed'] = batsman4['strikeMed'].item()
        sim2['earlyRat'] = batsman4['earlyRat'].item()
        sim2['midRat'] = batsman4['midRat'].item()
        sim2['lateRat'] = batsman4['lateRat'].item()
        sim2['aggression'] = batsman4['aggression'].item()
        sim2['aggSpin'] = batsman4['aggSpin'].item()
        sim2['aggSeam'] = batsman4['aggSeam'].item()
        sim2['aggMed'] = batsman4['aggMed'].item()
        sim2['passiveness'] = batsman4['passiveness'].item()
        sim2['pasSpin'] = batsman4['pasSpin'].item()
        sim2['pasSeam'] = batsman4['pasSeam'].item()
        sim2['pasMed'] = batsman4['pasMed'].item()
        temp = sim2['match.battingTeam.batsman.isRightHanded'].item()
        sim2['match.battingTeam.batsman.isRightHanded'] = sim2['match.battingTeam.batsmanPartner.isRightHanded'].item()
        sim2['match.battingTeam.batsmanPartner.isRightHanded'] = temp
    else:
        sim2['strikeRate'] = batsman3['strikeRate'].item()
        sim2['strikeSpin'] = batsman3['strikeSpin'].item()
        sim2['strikeSeam'] = batsman3['strikeSeam'].item()
        sim2['strikeMed'] = batsman3['strikeMed'].item()
        sim2['earlyRat'] = batsman3['earlyRat'].item()
        sim2['midRat'] = batsman3['midRat'].item()
        sim2['lateRat'] = batsman3['lateRat'].item()
        sim2['aggression'] = batsman3['aggression'].item()
        sim2['aggSpin'] = batsman3['aggSpin'].item()
        sim2['aggSeam'] = batsman3['aggSeam'].item()
        sim2['aggMed'] = batsman3['aggMed'].item()
        sim2['passiveness'] = batsman3['passiveness'].item()
        sim2['pasSpin'] = batsman3['pasSpin'].item()
        sim2['pasSeam'] = batsman3['pasSeam'].item()
        sim2['pasMed'] = batsman3['pasMed'].item()
print("Sim 2 yielded", totalRuns, "runs")

# Run simulation 3
totalRuns = 0
first = False
for i in range(6):
    runs = model.predict(sim3)
    probs = model.predict_proba(sim3)
    sim3['match.delivery.deliveryNumber.ball'] = sim3['match.delivery.deliveryNumber.ball'].item() + 1
    print(int(runs))
    print(probs)
    totalRuns += int(runs)
    if runs % 2 != 0:
        first = not (first)
    if first:
        sim3['strikeRate'] = batsman6['strikeRate'].item()
        sim3['strikeSpin'] = batsman6['strikeSpin'].item()
        sim3['strikeSeam'] = batsman6['strikeSeam'].item()
        sim3['strikeMed'] = batsman6['strikeMed'].item()
        sim3['earlyRat'] = batsman6['earlyRat'].item()
        sim3['midRat'] = batsman6['midRat'].item()
        sim3['lateRat'] = batsman6['lateRat'].item()
        sim3['aggression'] = batsman6['aggression'].item()
        sim3['aggSpin'] = batsman6['aggSpin'].item()
        sim3['aggSeam'] = batsman6['aggSeam'].item()
        sim3['aggMed'] = batsman6['aggMed'].item()
        sim3['passiveness'] = batsman6['passiveness'].item()
        sim3['pasSpin'] = batsman6['pasSpin'].item()
        sim3['pasSeam'] = batsman6['pasSeam'].item()
        sim3['pasMed'] = batsman6['pasMed'].item()
        temp = sim3['match.battingTeam.batsman.isRightHanded'].item()
        sim3['match.battingTeam.batsman.isRightHanded'] = sim3['match.battingTeam.batsmanPartner.isRightHanded'].item()
        sim3['match.battingTeam.batsmanPartner.isRightHanded'] = temp
    else:
        sim3['strikeRate'] = batsman5['strikeRate'].item()
        sim3['strikeSpin'] = batsman5['strikeSpin'].item()
        sim3['strikeSeam'] = batsman5['strikeSeam'].item()
        sim3['strikeMed'] = batsman5['strikeMed'].item()
        sim3['earlyRat'] = batsman5['earlyRat'].item()
        sim3['midRat'] = batsman5['midRat'].item()
        sim3['lateRat'] = batsman5['lateRat'].item()
        sim3['aggression'] = batsman5['aggression'].item()
        sim3['aggSpin'] = batsman5['aggSpin'].item()
        sim3['aggSeam'] = batsman5['aggSeam'].item()
        sim3['aggMed'] = batsman5['aggMed'].item()
        sim3['passiveness'] = batsman5['passiveness'].item()
        sim3['pasSpin'] = batsman5['pasSpin'].item()
        sim3['pasSeam'] = batsman5['pasSeam'].item()
        sim3['pasMed'] = batsman5['pasMed'].item()
print("Sim 3 yielded", totalRuns, "runs")

#Run simulation 4
totalRuns = 0
first = False
for i in range(6):
    runs = model.predict(sim4)
    probs = model.predict_proba(sim4)
    sim4['match.delivery.deliveryNumber.ball'] = sim4['match.delivery.deliveryNumber.ball'].item() + 1
    print(int(runs))
    print(probs)
    totalRuns += int(runs)
    if runs % 2 != 0:
        first = not (first)
    if first:
        sim4['strikeRate'] = batsman8['strikeRate'].item()
        sim4['strikeSpin'] = batsman8['strikeSpin'].item()
        sim4['strikeSeam'] = batsman8['strikeSeam'].item()
        sim4['strikeMed'] = batsman8['strikeMed'].item()
        sim4['earlyRat'] = batsman8['earlyRat'].item()
        sim4['midRat'] = batsman8['midRat'].item()
        sim4['lateRat'] = batsman8['lateRat'].item()
        sim4['aggression'] = batsman8['aggression'].item()
        sim4['aggSpin'] = batsman8['aggSpin'].item()
        sim4['aggSeam'] = batsman8['aggSeam'].item()
        sim4['aggMed'] = batsman8['aggMed'].item()
        sim4['passiveness'] = batsman8['passiveness'].item()
        sim4['pasSpin'] = batsman8['pasSpin'].item()
        sim4['pasSeam'] = batsman8['pasSeam'].item()
        sim4['pasMed'] = batsman8['pasMed'].item()
        temp = sim4['match.battingTeam.batsman.isRightHanded'].item()
        sim4['match.battingTeam.batsman.isRightHanded'] = sim4['match.battingTeam.batsmanPartner.isRightHanded'].item()
        sim4['match.battingTeam.batsmanPartner.isRightHanded'] = temp
    else:
        sim4['strikeRate'] = batsman7['strikeRate'].item()
        sim4['strikeSpin'] = batsman7['strikeSpin'].item()
        sim4['strikeSeam'] = batsman7['strikeSeam'].item()
        sim4['strikeMed'] = batsman7['strikeMed'].item()
        sim4['earlyRat'] = batsman7['earlyRat'].item()
        sim4['midRat'] = batsman7['midRat'].item()
        sim4['lateRat'] = batsman7['lateRat'].item()
        sim4['aggression'] = batsman7['aggression'].item()
        sim4['aggSpin'] = batsman7['aggSpin'].item()
        sim4['aggSeam'] = batsman7['aggSeam'].item()
        sim4['aggMed'] = batsman7['aggMed'].item()
        sim4['passiveness'] = batsman7['passiveness'].item()
        sim4['pasSpin'] = batsman7['pasSpin'].item()
        sim4['pasSeam'] = batsman7['pasSeam'].item()
        sim4['pasMed'] = batsman7['pasMed'].item()
print("Sim 4 yielded", totalRuns, "runs")

# Run simulation 5
totalRuns = 0
first = False
for i in range(6):
    runs = model.predict(sim5)
    probs = model.predict_proba(sim5)
    sim5['match.delivery.deliveryNumber.ball'] = sim5['match.delivery.deliveryNumber.ball'].item() + 1
    print(int(runs))
    print(probs)
    totalRuns += int(runs)
    if runs % 2 != 0:
        first = not (first)
    if first:
        sim5['strikeRate'] = batsman10['strikeRate'].item()
        sim5['strikeSpin'] = batsman10['strikeSpin'].item()
        sim5['strikeSeam'] = batsman10['strikeSeam'].item()
        sim5['strikeMed'] = batsman10['strikeMed'].item()
        sim5['earlyRat'] = batsman10['earlyRat'].item()
        sim5['midRat'] = batsman10['midRat'].item()
        sim5['lateRat'] = batsman10['lateRat'].item()
        sim5['aggression'] = batsman10['aggression'].item()
        sim5['aggSpin'] = batsman10['aggSpin'].item()
        sim5['aggSeam'] = batsman10['aggSeam'].item()
        sim5['aggMed'] = batsman10['aggMed'].item()
        sim5['passiveness'] = batsman10['passiveness'].item()
        sim5['pasSpin'] = batsman10['pasSpin'].item()
        sim5['pasSeam'] = batsman10['pasSeam'].item()
        sim5['pasMed'] = batsman10['pasMed'].item()
        temp = sim5['match.battingTeam.batsman.isRightHanded'].item()
        sim5['match.battingTeam.batsman.isRightHanded'] = sim5['match.battingTeam.batsmanPartner.isRightHanded'].item()
        sim5['match.battingTeam.batsmanPartner.isRightHanded'] = temp
    else:
        sim5['strikeRate'] = batsman9['strikeRate'].item()
        sim5['strikeSpin'] = batsman9['strikeSpin'].item()
        sim5['strikeSeam'] = batsman9['strikeSeam'].item()
        sim5['strikeMed'] = batsman9['strikeMed'].item()
        sim5['earlyRat'] = batsman9['earlyRat'].item()
        sim5['midRat'] = batsman9['midRat'].item()
        sim5['lateRat'] = batsman9['lateRat'].item()
        sim5['aggression'] = batsman9['aggression'].item()
        sim5['aggSpin'] = batsman9['aggSpin'].item()
        sim5['aggSeam'] = batsman9['aggSeam'].item()
        sim5['aggMed'] = batsman9['aggMed'].item()
        sim5['passiveness'] = batsman9['passiveness'].item()
        sim5['pasSpin'] = batsman9['pasSpin'].item()
        sim5['pasSeam'] = batsman9['pasSeam'].item()
        sim5['pasMed'] = batsman9['pasMed'].item()
print("Sim 5 yielded", totalRuns, "runs")

# Run simulation 6
totalRuns = 0
first = False
for i in range(6):
    probList = []
    runs = model.predict(sim6)
    probs = model.predict_proba(sim6)
    for i in probs[0]:
        probList.append(i)
    print(probs.shape)
    xAxis = ['-1', '0', '1', '2', '3', '4', '6']
    plt.bar(xAxis, probList)
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.xticks(xAxis, xAxis)
    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    # ax.set_ylabel('Probability')
    # ax.set_xlabel('Class')
    # ax.bar(xAxis, probList)
    plt.show()
    sim6['match.delivery.deliveryNumber.ball'] = sim6['match.delivery.deliveryNumber.ball'].item() + 1
    print(int(runs))
    print(probs)
    totalRuns += int(runs)
    if runs % 2 != 0:
        first = not (first)
    if first:
        sim6['strikeRate'] = batsman12['strikeRate'].item()
        sim6['strikeSpin'] = batsman12['strikeSpin'].item()
        sim6['strikeSeam'] = batsman12['strikeSeam'].item()
        sim6['strikeMed'] = batsman12['strikeMed'].item()
        sim6['earlyRat'] = batsman12['earlyRat'].item()
        sim6['midRat'] = batsman12['midRat'].item()
        sim6['lateRat'] = batsman12['lateRat'].item()
        sim6['aggression'] = batsman12['aggression'].item()
        sim6['aggSpin'] = batsman12['aggSpin'].item()
        sim6['aggSeam'] = batsman12['aggSeam'].item()
        sim6['aggMed'] = batsman12['aggMed'].item()
        sim6['passiveness'] = batsman12['passiveness'].item()
        sim6['pasSpin'] = batsman12['pasSpin'].item()
        sim6['pasSeam'] = batsman12['pasSeam'].item()
        sim6['pasMed'] = batsman12['pasMed'].item()
        temp = sim6['match.battingTeam.batsman.isRightHanded'].item()
        sim6['match.battingTeam.batsman.isRightHanded'] = sim6['match.battingTeam.batsmanPartner.isRightHanded'].item()
        sim6['match.battingTeam.batsmanPartner.isRightHanded'] = temp
    else:
        sim6['strikeRate'] = batsman11['strikeRate'].item()
        sim6['strikeSpin'] = batsman11['strikeSpin'].item()
        sim6['strikeSeam'] = batsman11['strikeSeam'].item()
        sim6['strikeMed'] = batsman11['strikeMed'].item()
        sim6['earlyRat'] = batsman11['earlyRat'].item()
        sim6['midRat'] = batsman11['midRat'].item()
        sim6['lateRat'] = batsman11['lateRat'].item()
        sim6['aggression'] = batsman11['aggression'].item()
        sim6['aggSpin'] = batsman11['aggSpin'].item()
        sim6['aggSeam'] = batsman11['aggSeam'].item()
        sim6['aggMed'] = batsman11['aggMed'].item()
        sim6['passiveness'] = batsman11['passiveness'].item()
        sim6['pasSpin'] = batsman11['pasSpin'].item()
        sim6['pasSeam'] = batsman11['pasSeam'].item()
        sim6['pasMed'] = batsman11['pasMed'].item()
print("Sim 6 yielded", totalRuns, "runs")