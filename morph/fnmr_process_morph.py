# %%
import pickle
import pandas as pd
import csv

results = list()

datasetFile = open("fnmr/dataset/fnmr_dataset_morph.pickle", 'rb')
dataset = pickle.load(datasetFile)

resultFile = open("fnmr/results/openface_fnmr_morph.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/results/arcface_fnmr_morph.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/results/facenet_fnmr_morph.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/results/vggface_fnmr_morph.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/results/faceapi_fnmr_morph.pickle", 'rb')
faceapiDataset = (pickle.load(resultFile))


resultDict = {
    "Facenet": dict(),
    "VGG-Face": dict(),
    "OpenFace": dict(),
    "ArcFace": dict()
}

for result in results:
    for algorithm in result.keys():
        if result[algorithm]:
            resultDict[algorithm] = result[algorithm]

# %%
labels = pd.read_csv(r'./labels_morph.csv')
names = list()
labelDict = dict()

for index, row in labels.iterrows():
    labelDict[str(row["id_num"])] = dict()
    labelDict[str(row["id_num"])]["Asian"] = False
    labelDict[str(row["id_num"])]["White"] = False
    labelDict[str(row["id_num"])]["Black"] = False
    labelDict[str(row["id_num"])]["Hispanic"] = False

    if row["race"] == "W":
        labelDict[str(row["id_num"])]["White"] = True
    elif row["race"] == "B":
        labelDict[str(row["id_num"])]["Black"] = True
    elif row["race"] == "A":
        labelDict[str(row["id_num"])]["Asian"] = True  
    elif row["race"] == "H":
        labelDict[str(row["id_num"])]["Hispanic"] = True

# %%
processDict = {
    "Facenet": dict(),
    "VGG-Face": dict(),
    "OpenFace": dict(),
    "ArcFace": dict()
}

for algorithm in resultDict.keys():
    for index in range(len(dataset)):
        referenceSubject = (dataset[index][0].split('_'))[0].lstrip("0")
        comparisonSubject = (dataset[index][1].split('_'))[0].lstrip("0")

        pairNumber = 'pair_' + str(index + 1)
        if not(referenceSubject in processDict[algorithm].keys()):
            processDict[algorithm][referenceSubject] = list()
        processDict[algorithm][referenceSubject].append(resultDict[algorithm][pairNumber])

faceAPIResult = dict()

for comparison in faceapiDataset:
    parsedComparison = comparison.split(" ")
    parsedComparison[0] = parsedComparison[0].replace("/home/ubuntu/Album2/", "")
    parsedComparison[1] = parsedComparison[1].replace("/home/ubuntu/Album2/", "")
    parsedComparison[2].replace("undefined", "1")
    try: 
        parsedComparison[2] = float(parsedComparison[2])
    except:
        parsedComparison[2] = float(1)

    
    referenceSubject = (parsedComparison[0].split('_'))[0].lstrip("0")
    comparisonSubject = (parsedComparison[1].split('_'))[0].lstrip("0")
    distance = parsedComparison[2]

    comparisonDict = dict()
    comparisonDict["distance"] = distance
    if distance >= float(.6):
        comparisonDict["verified"] = False
    else:
        comparisonDict["verified"] = True

    if not(referenceSubject in faceAPIResult.keys()):
        faceAPIResult[referenceSubject] = list()
    faceAPIResult[referenceSubject].append(comparisonDict)

processDict["FaceAPI"] = faceAPIResult

# %%
verifiedDict = {
    "Facenet": dict(),
    "VGG-Face": dict(),
    "OpenFace": dict(),
    "ArcFace": dict(),
    "FaceAPI": dict()
}


for algorithm in processDict.keys():
    for subject in processDict[algorithm].keys():
        for comparison in processDict[algorithm][subject]:
            if not(subject in verifiedDict[algorithm].keys()):
                verifiedDict[algorithm][subject] = list()
            verified = False
            if comparison["distance"] < .6:
                verified = True
            verifiedDict[algorithm][subject].append(verified)



# %%
verifiedStats = {
    "Facenet": dict(),
    "VGG-Face": dict(),
    "OpenFace": dict(),
    "ArcFace": dict(),
    "FaceAPI": dict()
}

for algorithm in verifiedDict.keys():
    for subject in verifiedDict[algorithm].keys():
        for comparison in verifiedDict[algorithm][subject]:
            if not(subject in verifiedStats[algorithm].keys()):
                verifiedStats[algorithm][subject] = dict()
                verifiedStats[algorithm][subject]["tpr"] = 0
                verifiedStats[algorithm][subject]["fnr"] = 0
            if comparison:
                verifiedStats[algorithm][subject]["tpr"] = verifiedStats[algorithm][subject]["tpr"] + 1
            else:
                verifiedStats[algorithm][subject]["fnr"] = verifiedStats[algorithm][subject]["fnr"] + 1

# %% [markdown]
# Overall FPR and TNR by algorithm:

# %%
summaryVerifiedStats = {
    "Facenet": dict(),
    "VGG-Face": dict(),
    "OpenFace": dict(),
    "ArcFace": dict(),
    "FaceAPI": dict()
}

for algorithm in verifiedStats.keys():

    summaryVerifiedStats[algorithm]["tpr"] = 0
    summaryVerifiedStats[algorithm]["fnr"] = 0

    for subject in verifiedStats[algorithm].keys():
        summaryVerifiedStats[algorithm]["tpr"] = verifiedStats[algorithm][subject]["tpr"] + summaryVerifiedStats[algorithm]["tpr"]
        summaryVerifiedStats[algorithm]["fnr"] = verifiedStats[algorithm][subject]["fnr"] + summaryVerifiedStats[algorithm]["fnr"]

print("FNMR Results:")
for algorithm in verifiedStats.keys():
    print()
    print(algorithm)
    print("FN:  " + str(summaryVerifiedStats[algorithm]["fnr"]))
    print("TP:  " + str(summaryVerifiedStats[algorithm]["tpr"]))  
    print("FNMR:  " + str(round(float(summaryVerifiedStats[algorithm]["fnr"])/float(summaryVerifiedStats[algorithm]["fnr"] + summaryVerifiedStats[algorithm]["tpr"]) * 100, 3)) + "%")  


# %%
summaryVerifiedStatsLabelled = {
    "Facenet": dict(),
    "VGG-Face": dict(),
    "OpenFace": dict(),
    "ArcFace": dict(),
    "FaceAPI": dict()
}


for algorithm in verifiedStats.keys():
    for label in labelDict[list(labelDict.keys())[0]].keys():
        if not(label in summaryVerifiedStatsLabelled[algorithm].keys()):
            summaryVerifiedStatsLabelled[algorithm][label] = dict()
        summaryVerifiedStatsLabelled[algorithm][label]["tpr"] = 0
        summaryVerifiedStatsLabelled[algorithm][label]["fnr"] = 0

    for subject in verifiedStats[algorithm].keys():
        labelledRace = ""
        for label in labelDict[subject].keys():
            if labelDict[subject][label]:
                labelledRace = label
        if labelledRace == "":
            continue
        summaryVerifiedStatsLabelled[algorithm][labelledRace]["tpr"] = verifiedStats[algorithm][subject]["tpr"] + summaryVerifiedStatsLabelled[algorithm][labelledRace]["tpr"]
        summaryVerifiedStatsLabelled[algorithm][labelledRace]["fnr"] = verifiedStats[algorithm][subject]["fnr"] + summaryVerifiedStatsLabelled[algorithm][labelledRace]["fnr"]


for algorithm in summaryVerifiedStatsLabelled.keys():
    for label in summaryVerifiedStatsLabelled[algorithm]:
        summaryVerifiedStatsLabelled[algorithm][label]["fnmr"] = str(round((float(summaryVerifiedStatsLabelled[algorithm][label]["fnr"])/float(summaryVerifiedStatsLabelled[algorithm][label]["fnr"] + summaryVerifiedStatsLabelled[algorithm][label]["tpr"]))*100,3)) + "%"

print("FNMR Results:")
for algorithm in verifiedStats.keys():
    print()
    print(algorithm)
    print("FN:  " + "\tBlack: " + str(summaryVerifiedStatsLabelled[algorithm]["Black"]["fnr"])+ "\tWhite: " + str(summaryVerifiedStatsLabelled[algorithm]["White"]["fnr"]) + "\tAsian: " + str(summaryVerifiedStatsLabelled[algorithm]["Asian"]["fnr"]) + "\tHispanic: " + str(summaryVerifiedStatsLabelled[algorithm]["Hispanic"]["fnr"]))
    print("TP:  " + "\tBlack: " + str(summaryVerifiedStatsLabelled[algorithm]["Black"]["tpr"])+ "\tWhite: " + str(summaryVerifiedStatsLabelled[algorithm]["White"]["tpr"]) + "\tAsian: " + str(summaryVerifiedStatsLabelled[algorithm]["Asian"]["tpr"]) + "\tHispanic: " + str(summaryVerifiedStatsLabelled[algorithm]["Hispanic"]["tpr"]))  
    print("FNMR:  " + "\tBlack: " + str(summaryVerifiedStatsLabelled[algorithm]["Black"]["fnmr"])+ "\tWhite: " + str(summaryVerifiedStatsLabelled[algorithm]["White"]["fnmr"]) + "\tAsian: " + str(summaryVerifiedStatsLabelled[algorithm]["Asian"]["fnmr"]) + "\tHispanic: " + str(summaryVerifiedStatsLabelled[algorithm]["Hispanic"]["fnmr"]))




