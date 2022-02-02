# %%
import pickle
import pandas as pd
import csv

results = list()

datasetFile = open("fnmr/pose/dataset/fnmr_dataset_multipie_pose.pickle", 'rb')
dataset = pickle.load(datasetFile)

resultFile = open("fnmr/pose/results/openface_fnmr_multipie_pose.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/pose/results/arcface_fnmr_multipie_pose.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/pose/results/facenet_fnmr_multipie_pose.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/pose/results/vggface_fnmr_multipie_pose.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/pose/results/faceapi_fnmr_multipie_pose.pickle", 'rb')
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

        pairNumber = 'pair_' + str(index + 1)
        if not(referenceSubject in processDict[algorithm].keys()):
            processDict[algorithm][referenceSubject] = list()

        processDict[algorithm][referenceSubject].append(resultDict[algorithm][pairNumber])

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
        if not(subject in verifiedDict[algorithm].keys()):
            verifiedDict[algorithm][subject] = list()
        for comparison in processDict[algorithm][subject]:
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
    verifiedStats[algorithm]["tpr"] = 0
    verifiedStats[algorithm]["fnr"] = 0
    for subject in verifiedDict[algorithm].keys():
        for comparison in verifiedDict[algorithm][subject]:
            if comparison:
                verifiedStats[algorithm]["tpr"] = verifiedStats[algorithm]["tpr"] + 1
            else:
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



