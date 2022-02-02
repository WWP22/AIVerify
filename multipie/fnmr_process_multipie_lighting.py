# %%
import pickle
import pandas as pd
import csv

results = list()

datasetFile = open("fnmr/lighting/dataset/fnmr_dataset_multipie_lighting.pickle", 'rb')
dataset = pickle.load(datasetFile)

resultFile = open("fnmr/lighting/results/openface_fnmr_multipie_lighting.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/lighting/results/arcface_fnmr_multipie_lighting.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/lighting/results/facenet_fnmr_multipie_lighting.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/lighting/results/vggface_fnmr_multipie_lighting.pickle", 'rb')
results.append(pickle.load(resultFile))

resultFile = open("fnmr/lighting/results/faceapi_fnmr_multipie_lighting.pickle", 'rb')
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
        lighting = (dataset[index][1].split('/'))[6].split('_')[4].replace(".png", "")
        
        pairNumber = 'pair_' + str(index + 1)
        if not(referenceSubject in processDict[algorithm].keys()):
            processDict[algorithm][referenceSubject] = dict()
        if not(lighting in processDict[algorithm][referenceSubject].keys()):
            processDict[algorithm][referenceSubject][lighting] = list()
        processDict[algorithm][referenceSubject][lighting].append(resultDict[algorithm][pairNumber])
    
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
    lighting = (parsedComparison[1].split('/'))[6].split('_')[4].replace(".png", "")
    distance = parsedComparison[2]

    comparisonDict = dict()
    comparisonDict["distance"] = distance
    if distance >= float(.6):
        comparisonDict["verified"] = False
    else:
        comparisonDict["verified"] = True

    if not(referenceSubject in faceAPIResult.keys()):
        faceAPIResult[referenceSubject] = dict()
    if not(lighting in faceAPIResult[referenceSubject].keys()):
        faceAPIResult[referenceSubject][lighting] = list()

    faceAPIResult[referenceSubject][lighting].append(comparisonDict)

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
            verifiedDict[algorithm][subject] = dict()
        for lighting in processDict[algorithm][subject].keys():
            if not(lighting in verifiedDict[algorithm][subject].keys()):
                verifiedDict[algorithm][subject][lighting] = list()
            for comparison in processDict[algorithm][subject][lighting]:
                verified = False
                if comparison["distance"] < .6:
                    verified = True
                verifiedDict[algorithm][subject][lighting].append(verified)



# %%
verifiedStats = {
    "Facenet": dict(),
    "VGG-Face": dict(),
    "OpenFace": dict(),
    "ArcFace": dict(),
    "FaceAPI": dict()
}

lightingConditions = ["00", "04", "12", "16"]

for algorithm in verifiedDict.keys():
    verifiedStats[algorithm]["tpr"] = 0
    verifiedStats[algorithm]["fnr"] = 0
    verifiedStats[algorithm]["lighting"] = dict()
    for lightingCondition in lightingConditions:
        verifiedStats[algorithm]["lighting"][lightingCondition] = dict()
        verifiedStats[algorithm]["lighting"][lightingCondition]["tpr"] = 0
        verifiedStats[algorithm]["lighting"][lightingCondition]["fnr"] = 0
    for subject in verifiedDict[algorithm].keys():
        for lighting in verifiedDict[algorithm][subject].keys():
            for comparison in verifiedDict[algorithm][subject][lighting]:
                if comparison:
                    verifiedStats[algorithm]["lighting"][lighting]["tpr"] = verifiedStats[algorithm]["lighting"][lighting]["tpr"] + 1
                    verifiedStats[algorithm]["tpr"] = verifiedStats[algorithm]["tpr"] + 1
                else:
                    verifiedStats[algorithm]["lighting"][lighting]["fnr"] = verifiedStats[algorithm]["lighting"][lighting]["fnr"] + 1
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
    for lighting in verifiedStats[algorithm]["lighting"].keys():
        verifiedStats[algorithm]["lighting"][lighting]["fnmr"] = round((float(verifiedStats[algorithm]["lighting"][lighting]["fnr"])/(float(verifiedStats[algorithm]["lighting"][lighting]["fnr"] + verifiedStats[algorithm]["lighting"][lighting]["tpr"]))) * 100, 3)


print("FNMR Results:")
for algorithm in verifiedStats.keys():
    print()
    print(algorithm)
    print("FN:  " + "\t00: " + str(verifiedStats[algorithm]["lighting"]["00"]["fnr"]) + "\t04: " + str(verifiedStats[algorithm]["lighting"]["04"]["fnr"]) + "\t12: " + str(verifiedStats[algorithm]["lighting"]["12"]["fnr"]) + "\t16: " + str(verifiedStats[algorithm]["lighting"]["16"]["fnr"]))
    print("TP:  " + "\t00: " + str(verifiedStats[algorithm]["lighting"]["00"]["tpr"]) + "\t04: " + str(verifiedStats[algorithm]["lighting"]["04"]["tpr"]) + "\t12: " + str(verifiedStats[algorithm]["lighting"]["12"]["tpr"]) + "\t16: " + str(verifiedStats[algorithm]["lighting"]["16"]["tpr"]))
    print("FNMR:  " + "\t00: " + str(verifiedStats[algorithm]["lighting"]["00"]["fnmr"]) + "%\t04: " + str(verifiedStats[algorithm]["lighting"]["04"]["fnmr"]) + "%\t12: " + str(verifiedStats[algorithm]["lighting"]["12"]["fnmr"]) + "%\t16: " + str(verifiedStats[algorithm]["lighting"]["16"]["fnmr"]) + "%")


