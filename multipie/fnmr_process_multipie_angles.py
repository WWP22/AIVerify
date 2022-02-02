# %%
import pickle
import pandas as pd
import csv

results = list()

datasetFile = open("fnmr/angles/dataset/fnmr_dataset_multipie_angles.pickle", 'rb')
dataset = pickle.load(datasetFile)

resultFile = open("fnmr/angles/results/openface_fnmr_multipie_angles.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/angles/results/arcface_fnmr_multipie_angles.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/angles/results/facenet_fnmr_multipie_angles.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/angles/results/vggface_fnmr_multipie_angles.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/angles/results/faceapi_fnmr_multipie_angles.pickle", 'rb')
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
        angle = (dataset[index][1].split('/'))[5]

        pairNumber = 'pair_' + str(index + 1)
        if not(referenceSubject in processDict[algorithm].keys()):
            processDict[algorithm][referenceSubject] = dict()
        if not(angle in processDict[algorithm][referenceSubject].keys()):
            processDict[algorithm][referenceSubject][angle] = list()
        processDict[algorithm][referenceSubject][angle].append(resultDict[algorithm][pairNumber])

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
    angle = (parsedComparison[1].split('/'))[5]
    distance = parsedComparison[2]

    comparisonDict = dict()
    comparisonDict["distance"] = distance
    if distance >= float(.6):
        comparisonDict["verified"] = False
    else:
        comparisonDict["verified"] = True

    if not(referenceSubject in faceAPIResult.keys()):
        faceAPIResult[referenceSubject] = dict()
    if not(angle in faceAPIResult[referenceSubject].keys()):
        faceAPIResult[referenceSubject][angle] = list()

    faceAPIResult[referenceSubject][angle].append(comparisonDict)

processDict["FaceAPI"] = faceAPIResult


# %%
verifiedDict = {
    "Facenet": dict(),
    "VGG-Face": dict(),
    "OpenFace": dict(),
    "ArcFace": dict(),
    "FaceAPI": dict()
}

angleNames = {
    "11_0": "-90",
    "12_0": "-75",
    "09_0": "-60",
    "08_0": "-45",
    "13_0": "-30",
    "14_0": "-15",
    "05_1": "0",
    "05_0": "15",
    "04_1": "30",
    "19_0": "45",
    "20_0": "60",
    "01_0": "75",
    "24_0": "90"
}

for algorithm in processDict.keys():
    for subject in processDict[algorithm].keys():
        if not(subject in verifiedDict[algorithm].keys()):
            verifiedDict[algorithm][subject] = dict()
        for angle in processDict[algorithm][subject].keys():
            if not(angleNames[angle] in verifiedDict[algorithm][subject].keys()):
                verifiedDict[algorithm][subject][angleNames[angle]] = list()
            for comparison in processDict[algorithm][subject][angle]:
                verified = False
                if comparison["distance"] < .6:
                    verified = True
                verifiedDict[algorithm][subject][angleNames[angle]].append(verified)



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
    verifiedStats[algorithm]["angle"] = dict()
    for angle in angleNames.keys():
        if angle == "05_1":
            continue
        verifiedStats[algorithm]["angle"][angleNames[angle]] = dict()
        verifiedStats[algorithm]["angle"][angleNames[angle]]["tpr"] = 0
        verifiedStats[algorithm]["angle"][angleNames[angle]]["fnr"] = 0
    for subject in verifiedDict[algorithm].keys():
        for angle in verifiedDict[algorithm][subject].keys():
            for comparison in verifiedDict[algorithm][subject][angle]:
                if comparison:
                    verifiedStats[algorithm]["angle"][angle]["tpr"] = verifiedStats[algorithm]["angle"][angle]["tpr"] + 1
                    verifiedStats[algorithm]["tpr"] = verifiedStats[algorithm]["tpr"] + 1
                else:
                    verifiedStats[algorithm]["angle"][angle]["fnr"] = verifiedStats[algorithm]["angle"][angle]["fnr"] + 1
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
    for angle in verifiedStats[algorithm]["angle"].keys():
        verifiedStats[algorithm]["angle"][angle]["fnmr"] = round((float(verifiedStats[algorithm]["angle"][angle]["fnr"])/(float(verifiedStats[algorithm]["angle"][angle]["fnr"] + verifiedStats[algorithm]["angle"][angle]["tpr"]))) * 100, 3)


print("FNMR Results:")
for algorithm in verifiedStats.keys():
    print()
    print(algorithm)
    print("FN:  " + "\t-90: " + str(verifiedStats[algorithm]["angle"]["-90"]["fnr"]) + "\t-75: " + str(verifiedStats[algorithm]["angle"]["-75"]["fnr"]) + "\t-60: " + str(verifiedStats[algorithm]["angle"]["-60"]["fnr"]) + "\t-45: " + str(verifiedStats[algorithm]["angle"]["-45"]["fnr"]) + "\t-30: " + str(verifiedStats[algorithm]["angle"]["-30"]["fnr"]) + "\t-15: " + str(verifiedStats[algorithm]["angle"]["-15"]["fnr"]) + "\t15: " + str(verifiedStats[algorithm]["angle"]["15"]["fnr"]) + "\t30: " + str(verifiedStats[algorithm]["angle"]["30"]["fnr"]) + "\t45: " + str(verifiedStats[algorithm]["angle"]["45"]["fnr"]) + "\t60: " + str(verifiedStats[algorithm]["angle"]["60"]["fnr"]) + "\t75: " + str(verifiedStats[algorithm]["angle"]["75"]["fnr"]) + "\t90: " + str(verifiedStats[algorithm]["angle"]["90"]["fnr"]))
    print("TP:  " + "\t-90: " + str(verifiedStats[algorithm]["angle"]["-90"]["tpr"]) + "\t-75: " + str(verifiedStats[algorithm]["angle"]["-75"]["tpr"]) + "\t-60: " + str(verifiedStats[algorithm]["angle"]["-60"]["tpr"]) + "\t-45: " + str(verifiedStats[algorithm]["angle"]["-45"]["tpr"]) + "\t-30: " + str(verifiedStats[algorithm]["angle"]["-30"]["tpr"]) + "\t-15: " + str(verifiedStats[algorithm]["angle"]["-15"]["tpr"]) + "\t15: " + str(verifiedStats[algorithm]["angle"]["15"]["tpr"]) + "\t30: " + str(verifiedStats[algorithm]["angle"]["30"]["tpr"]) + "\t45: " + str(verifiedStats[algorithm]["angle"]["45"]["tpr"]) + "\t60: " + str(verifiedStats[algorithm]["angle"]["60"]["tpr"]) + "\t75: " + str(verifiedStats[algorithm]["angle"]["75"]["tpr"]) + "\t90: " + str(verifiedStats[algorithm]["angle"]["90"]["tpr"]))
    print("FNMR:  " + "\t-90: " + str(verifiedStats[algorithm]["angle"]["-90"]["fnmr"]) + "%\t-75: " + str(verifiedStats[algorithm]["angle"]["-75"]["fnmr"]) + "%\t-60: " + str(verifiedStats[algorithm]["angle"]["-60"]["fnmr"]) + "%\t-45: " + str(verifiedStats[algorithm]["angle"]["-45"]["fnmr"]) + "%\t-30: " + str(verifiedStats[algorithm]["angle"]["-30"]["fnmr"]) + "%\t-15: " + str(verifiedStats[algorithm]["angle"]["-15"]["fnmr"]) + "%\t15: " + str(verifiedStats[algorithm]["angle"]["15"]["fnmr"]) + "%\t30: " + str(verifiedStats[algorithm]["angle"]["30"]["fnmr"]) + "%\t45: " + str(verifiedStats[algorithm]["angle"]["45"]["fnmr"]) + "%\t60: " + str(verifiedStats[algorithm]["angle"]["60"]["fnmr"]) + "%\t75: " + str(verifiedStats[algorithm]["angle"]["75"]["fnmr"]) + "%\t90: " + str(verifiedStats[algorithm]["angle"]["90"]["fnmr"]) + "%")


