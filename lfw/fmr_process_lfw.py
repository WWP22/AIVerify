# %%
import pickle
import pandas as pd
import csv

results = list()

datasetFile = open("fmr/dataset/fmr_dataset_lfw.pickle", 'rb')
dataset = pickle.load(datasetFile)

resultFile = open("fmr/results/openface_fmr_lfw.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fmr/results/arcface_fmr_lfw.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fmr/results/facenet_fmr_lfw.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fmr/results/vggface_fmr_lfw.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fmr/results/faceapi_fmr_lfw.pickle", 'rb')
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
labels = pd.read_csv(r'./labels_lfw.csv')
names = list()
labelDict = dict()

with open('names_lfw.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)[0]
    for name in data:
        names.append(name.split('\\')[0])

for index, row in labels.iterrows():
    labelDict[names[index]] = dict()
    labelDict[names[index]]["Asian"] = bool(row["Asian"])
    labelDict[names[index]]["White"] = bool(row["White"])
    labelDict[names[index]]["Black"] = bool(row["Black"])
    labelDict[names[index]]["Indian"] = bool(row["Indian"])

# %%
processDict = {
    "Facenet": dict(),
    "VGG-Face": dict(),
    "OpenFace": dict(),
    "ArcFace": dict()
}

for algorithm in resultDict.keys():
    for index in range(len(dataset)):
        referenceSubject = (dataset[index][0].split('/'))[0]
        comparisonSubject = (dataset[index][1].split('/'))[0]

        pairNumber = 'pair_' + str(index + 1)
        if not(referenceSubject in processDict[algorithm].keys()):
            processDict[algorithm][referenceSubject] = list()
        resultDict[algorithm][pairNumber]["comparison_subject"] = comparisonSubject
        processDict[algorithm][referenceSubject].append(resultDict[algorithm][pairNumber])

faceAPIResult = dict()

for comparison in faceapiDataset:
    parsedComparison = comparison.split(" ")
    parsedComparison[0] = parsedComparison[0].replace("/home/ubuntu/lfw/", "")
    parsedComparison[1] = parsedComparison[1].replace("/home/ubuntu/lfw/", "")
    parsedComparison[2].replace("undefined", "1")
    try: 
        parsedComparison[2] = float(parsedComparison[2])
    except:
        parsedComparison[2] = float(1)

    
    referenceSubject = (parsedComparison[0].split('/'))[0]
    comparisonSubject = (parsedComparison[1].split('/'))[0]
    distance = parsedComparison[2]

    comparisonDict = dict()
    comparisonDict["distance"] = distance
    comparisonDict["comparison_subject"] = comparisonSubject
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
                verifiedStats[algorithm][subject]["tnr"] = 0
                verifiedStats[algorithm][subject]["fpr"] = 0
            if comparison:
                verifiedStats[algorithm][subject]["fpr"] = verifiedStats[algorithm][subject]["fpr"] + 1
            else:
                verifiedStats[algorithm][subject]["tnr"] = verifiedStats[algorithm][subject]["tnr"] + 1

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

    summaryVerifiedStats[algorithm]["fpr"] = 0
    summaryVerifiedStats[algorithm]["tnr"] = 0

    for subject in verifiedStats[algorithm].keys():
        summaryVerifiedStats[algorithm]["fpr"] = verifiedStats[algorithm][subject]["fpr"] + summaryVerifiedStats[algorithm]["fpr"]
        summaryVerifiedStats[algorithm]["tnr"] = verifiedStats[algorithm][subject]["tnr"] + summaryVerifiedStats[algorithm]["tnr"]

print("FMR Results:")
for algorithm in verifiedStats.keys():
    print()
    print(algorithm)
    print("FP:  " + str(summaryVerifiedStats[algorithm]["fpr"]))
    print("TN:  " + str(summaryVerifiedStats[algorithm]["tnr"]))  
    print("FMR:  " + str(round(float(summaryVerifiedStats[algorithm]["fpr"])/float(summaryVerifiedStats[algorithm]["fpr"] + summaryVerifiedStats[algorithm]["tnr"]) * 100, 3)) + "%")  


# %%



