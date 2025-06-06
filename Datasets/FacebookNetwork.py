import numpy as np

VAFile = "VAZipcodesLatLong.csv"
FBFile = "VirginiaZipCodeLinks.csv"

def generateVALocations(file):
    # Initializes ZIP codes and latitude and longitude coordinates
    zipCodeToNode = {}
    locations = []
    zcIdx = 0
    with open (file) as file1:
        locList = [line.strip() for line in file1]
    for loc in locList[1:]:
        loc = loc.split(",")
        locations.append((float(loc[2]), float(loc[1])))
        zipCodeToNode[int(loc[0])] = zcIdx
        zcIdx+=1
    pop = zcIdx
    return pop, locations, zipCodeToNode

def generateFaceBookNetwork(file, zipCodeToNode, k):
    # Generates social network from Social Connectedness Index (SCI)
    # Links each ZIP code to the k ZIP codes it has the highest SCIs with
    nearestNeighbors = {}
    with open (file) as file1:
        locList = [line.strip() for line in file1]
    for loc in locList[1:]:
        fromLoc, toLoc, sci = list(map(int, loc.split(",")))
        if fromLoc == toLoc:
            continue
        fromID = zipCodeToNode[fromLoc]
        toID = zipCodeToNode[toLoc]
        if fromID not in nearestNeighbors:
            nearestNeighbors[fromID] = [(sci, toID)]
        elif nearestNeighbors[fromID][-1][0] > sci:
            if len(nearestNeighbors[fromID]) < k:
                nearestNeighbors[fromID].append((sci, toID))
        else:
            i = len(nearestNeighbors[fromID])-1
            while nearestNeighbors[fromID][i][0] < sci and i > -1:
                i-=1
            nearestNeighbors[fromID].insert(i+1, (sci, toID))
            if len(nearestNeighbors[fromID]) > k:
                nearestNeighbors[fromID].pop()
    return nearestNeighbors


def FacebookNetwork():
    # Generates matrix and locations for the Facebook Location Data
    k = 10
    pop, locations, zipCodeToNode = generateVALocations(VAFile)
    nearestNeighbors = generateFaceBookNetwork(FBFile, zipCodeToNode, k) 
    matrix = np.zeros((pop, pop))
    for fromID in nearestNeighbors:
        for toID in nearestNeighbors[fromID]:
            toID = toID[1]
            matrix[fromID, toID] = 1
            matrix[toID, fromID] = 1
    return matrix, locations