# %%
import pickle
import pandas as pd
import csv

results = list()

datasetFile = open("fnmr/age/dataset/fnmr_dataset_multipie_age.pickle", 'rb')
dataset = pickle.load(datasetFile)

resultFile = open("fnmr/age/results/openface_fnmr_multipie_age.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/age/results/arcface_fnmr_multipie_age.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/age/results/facenet_fnmr_multipie_age.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/age/results/vggface_fnmr_multipie_age.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/age/results/faceapi_fnmr_multipie_age.pickle", 'rb')
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
processDict = {
    "Facenet": dict(),
    "VGG-Face": dict(),
    "OpenFace": dict(),
    "ArcFace": dict()
}

for algorithm in resultDict.keys():
    for index in range(len(dataset)):
        referenceSubject = (dataset[index][0].split('/'))[3]
        session = (dataset[index][1].split('/'))[1]

        pairNumber = 'pair_' + str(index + 1)
        if not(referenceSubject in processDict[algorithm].keys()):
            processDict[algorithm][referenceSubject] = dict()
        if not(session in processDict[algorithm][referenceSubject].keys()):
            processDict[algorithm][referenceSubject][session] = list()
        resultDict[algorithm][pairNumber]["referenceSession"] = (dataset[index][0].split('/'))[1]
        processDict[algorithm][referenceSubject][session].append(resultDict[algorithm][pairNumber])

faceAPIResult = dict()

for comparison in faceapiDataset:
    parsedComparison = comparison.split(" ")
    parsedComparison[0] = parsedComparison[0].replace("/home/ubuntu/Multipie/Multi-Pie/", "")
    parsedComparison[1] = parsedComparison[1].replace("/home/ubuntu/Multipie/Multi-Pie/", "")
    parsedComparison[2].replace("undefined", "1")
    try: 
        parsedComparison[2] = float(parsedComparison[2])
    except:
        parsedComparison[2] = float(1)

    
    referenceSubject = (parsedComparison[0].split('/'))[3]
    session = (parsedComparison[1].split('/'))[1]
    referenceSession = (parsedComparison[0].split('/'))[1]
    distance = parsedComparison[2]

    comparisonDict = dict()
    comparisonDict["referenceSession"] = referenceSession
    comparisonDict["distance"] = distance
    if distance >= float(.6):
        comparisonDict["verified"] = False
    else:
        comparisonDict["verified"] = True

    if not(referenceSubject in faceAPIResult.keys()):
        faceAPIResult[referenceSubject] = dict()
    if not(session in faceAPIResult[referenceSubject].keys()):
        faceAPIResult[referenceSubject][session] = list()

    faceAPIResult[referenceSubject][session].append(comparisonDict)

processDict["FaceAPI"] = faceAPIResult


# %%


# %%
verifiedDict = {
    "Facenet": dict(),
    "VGG-Face": dict(),
    "OpenFace": dict(),
    "ArcFace": dict(),
    "FaceAPI": dict()
}

sessionGaps = [1, 2, 3]

for algorithm in processDict.keys():
    for subject in processDict[algorithm].keys():
        if not(subject in verifiedDict[algorithm].keys()):
            verifiedDict[algorithm][subject] = dict()
            for gap in sessionGaps:
                verifiedDict[algorithm][subject][gap] = list()
        for session in processDict[algorithm][subject].keys():
            for comparison in processDict[algorithm][subject][session]:
                verified = False
                if comparison["distance"] < .6:
                    verified = True
                sessionInt = int(session.replace("session0", ""))
                referenceInt = int(comparison["referenceSession"].replace("session0", ""))
                gap = abs(sessionInt - referenceInt)
                verifiedDict[algorithm][subject][gap].append(verified)



# %%
verifiedStats = {
    "Facenet": dict(),
    "VGG-Face": dict(),
    "OpenFace": dict(),
    "ArcFace": dict(),
    "FaceAPI": dict()
}

for algorithm in verifiedDict.keys():
    verifiedStats[algorithm]["tpr"] = 0
    verifiedStats[algorithm]["fnr"] = 0
    verifiedStats[algorithm]["gap"] = dict()
    for sessionGap in sessionGaps:
        verifiedStats[algorithm]["gap"][sessionGap] = dict()
        verifiedStats[algorithm]["gap"][sessionGap]["tpr"] = 0
        verifiedStats[algorithm]["gap"][sessionGap]["fnr"] = 0
    for subject in verifiedDict[algorithm].keys():
        for gap in verifiedDict[algorithm][subject].keys():
            for comparison in verifiedDict[algorithm][subject][gap]:
                if comparison:
                    verifiedStats[algorithm]["gap"][gap]["tpr"] = verifiedStats[algorithm]["gap"][gap]["tpr"] + 1
                    verifiedStats[algorithm]["tpr"] = verifiedStats[algorithm]["tpr"] + 1
                else:
                    verifiedStats[algorithm]["gap"][gap]["fnr"] = verifiedStats[algorithm]["gap"][gap]["fnr"] + 1
                    verifiedStats[algorithm]["fnr"] = verifiedStats[algorithm]["fnr"] + 1

# %% [markdown]
# Overall FPR and TNR by algorithm:

# %%
print("FNMR Results:")
for algorithm in verifiedStats.keys():
    print()
    print(algorithm)
    print("FN:  " + str(verifiedStats[algorithm]["fnr"]))
    print("TP:  " + str(verifiedStats[algorithm]["tpr"]))  
    print("FNMR:  " + str(round(float(verifiedStats[algorithm]["fnr"])/float(verifiedStats[algorithm]["fnr"] + verifiedStats[algorithm]["tpr"]) * 100, 3)) + "%")  


# %%
for algorithm in verifiedStats.keys():
    for gap in verifiedStats[algorithm]["gap"].keys():
        verifiedStats[algorithm]["gap"][gap]["fnmr"] = round((float(verifiedStats[algorithm]["gap"][gap]["fnr"])/(float(verifiedStats[algorithm]["gap"][gap]["fnr"] + verifiedStats[algorithm]["gap"][gap]["tpr"]))) * 100, 3)


print("FNMR Results:")
for algorithm in verifiedStats.keys():
    print()
    print(algorithm)
    print("FN:  " + "\tGap_1: " + str(verifiedStats[algorithm]["gap"][1]["fnr"]) + "\tGap_2: " + str(verifiedStats[algorithm]["gap"][2]["fnr"]) + "\tGap_3: " + str(verifiedStats[algorithm]["gap"][3]["fnr"]))
    print("TP:  " + "\tGap_1: " + str(verifiedStats[algorithm]["gap"][1]["tpr"]) + "\tGap_2: " + str(verifiedStats[algorithm]["gap"][2]["tpr"]) + "\tGap_3: " + str(verifiedStats[algorithm]["gap"][3]["tpr"]))
    print("FNMR:  " +  "\tGap_1: " + str(verifiedStats[algorithm]["gap"][1]["fnmr"]) + "%\tGap_2: " + str(verifiedStats[algorithm]["gap"][2]["fnmr"]) + "%\tGap_3: " + str(verifiedStats[algorithm]["gap"][3]["fnmr"]) + "%")



