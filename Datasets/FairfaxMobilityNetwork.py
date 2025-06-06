import numpy as np

locFile = "FairfaxCensusTractLatLong.csv"
GTFile = "FairfaxCensusTracts.csv"

def countyInit(file):
    # Initializes census tracts and latitude and longitude coordinates
    seenSet = set()
    tractToNode = {}
    locations = []
    tIdx = 0
    with open (file) as file1:
        locList = [line.strip() for line in file1]
    for loc in locList[1:]:
        loc = loc.split(",")
        tract = loc[0]
        if tract not in seenSet:
            locations.append((float(loc[2]), float(loc[1])))
            tractToNode[int(loc[0])] = tIdx
            tIdx+=1
            seenSet.add(tract)
    pop = tIdx
    return pop, locations, tractToNode

def getNearestNodes(GTFile, tractToNode, p):
    # Generates social network from mobility data
    # Links census tracts with the top p population flows between them
    nearestNeighbors = {}
    with open (GTFile) as file1:
        locList = [line.strip() for line in file1]
    sortedFlows = []
    locsToFlow = {}
    for loc in locList[1:]:
        toFromList = loc.split(",")
        fromLoc, toLoc, popFlows = int(toFromList[0]), int(toFromList[1]), float(toFromList[8])
        if fromLoc == toLoc:
            continue
        fromID = tractToNode[fromLoc]
        if fromLoc not in nearestNeighbors:
            nearestNeighbors[fromID] = set()
        sortedFlows.append(popFlows)
        locsToFlow[(fromLoc, toLoc)] = popFlows
    sortedFlows.sort()
    minFlow = sortedFlows[int(len(sortedFlows)*p)]
    for locs in locsToFlow:
        if locsToFlow[locs] >= minFlow:
            loc1 = tractToNode[locs[0]]
            loc2 = tractToNode[locs[1]]
            nearestNeighbors[loc1].add(loc2)
            nearestNeighbors[loc2].add(loc1)
    return nearestNeighbors

def FairfaxNetwork():
    # Generates matrix and locations for the Fairfax Network
    p = 0.87
    pop, locations, tractToNode = countyInit(locFile)
    nearestNeighbors = getNearestNodes(GTFile, tractToNode, p) 
    matrix = np.zeros((pop, pop))
    for fromID in nearestNeighbors:
        for toID in nearestNeighbors[fromID]:
            matrix[fromID, toID] = 1
            matrix[toID, fromID] = 1
    return matrix, locations