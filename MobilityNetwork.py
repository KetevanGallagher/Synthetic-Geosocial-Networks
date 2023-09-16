import matplotlib.pyplot as plt
import random as rd
import numpy as np
import networkx as nx
import time
import statistics

VAFile = "FairfaxCensusTractLatLong.csv"
GTFile = "FairfaxCensusTracts.csv"

def countyInit(file):
    seenSet = set()
    global agents, pop
    agents = {}
    agentIdx = 0
    with open (file) as file1:
        countyList = [line.strip() for line in file1]
    for county in countyList[1:]:
        county = county.split(",")
        tract = county[0]
        if tract not in seenSet:
            agents[int(tract)] = (float(county[2]), float(county[1]))
            agentIdx+=1
            seenSet.add(tract)
    pop = agentIdx

def getNearestNodes(k, locFile):
    global nearestNeighbors
    nearestNeighbors = {}
    with open (locFile) as file1:
        locList = [line.strip() for line in file1]
    for loc in locList[1:]:
        toFromList = loc.split(",")
        fromLoc, toLoc, popFlows = int(toFromList[0]), int(toFromList[1]), float(toFromList[8])
        if fromLoc == toLoc:
            continue
        if fromLoc not in nearestNeighbors:
            nearestNeighbors[fromLoc] = [(popFlows, toLoc)]
        elif nearestNeighbors[fromLoc][-1][0] > popFlows:
            if len(nearestNeighbors[fromLoc]) < k:
                nearestNeighbors[fromLoc].append((popFlows, toLoc))
        else:
            i = len(nearestNeighbors[fromLoc])-1
            while nearestNeighbors[fromLoc][i][0] < popFlows and i > -1:
                i-=1
            nearestNeighbors[fromLoc].insert(i+1, (popFlows, toLoc))
            if len(nearestNeighbors[fromLoc]) > k:
                nearestNeighbors[fromLoc].pop()

def createGraph():
    network = nx.Graph()
    for agent in agents:
        coord = agents[agent]
        network.add_node(agent, pos=(coord[0], coord[1]))
    for fromLoc in nearestNeighbors:
        for toLoc in nearestNeighbors[fromLoc]:
            network.add_edge(fromLoc,toLoc[1])
            network.add_edge(toLoc[1],fromLoc)
    return network


start_time = time.time()

k = 8

countyInit(VAFile)
getNearestNodes(k, GTFile)
mbG = createGraph()


degrees = mbG.degree()
sum_of_edges = sum(dict(degrees).values())
print("Average Degree:")
print(sum_of_edges/(pop))
print("Standard Deviation of the Degree:")
print(statistics.stdev(dict(degrees).values()))

avgPathLength = nx.average_shortest_path_length(mbG)

degList = [i for i in degrees]
maxDeg = max(degList, key= lambda x: x[1])
maximumDegree = maxDeg[1]


averageDistances = []
radiusOfGyration = []

for node in mbG:
    maxDistance = 0
    for n in mbG.neighbors(node):
        coord1, coord2 = agents[node], agents[n]
        dist = ((111139*float(coord1[0] - coord2[0]))**2 + ((111139*float(coord1[1] - coord2[1])**2)))**0.5
        averageDistances.append(dist)
        if dist > maxDistance:
            maxDistance = dist
    radiusOfGyration.append(maxDistance)

print("Radius of Gyration:")
print(sum(radiusOfGyration)/(len(agents)))
print("Standard Deviation of Radius of Gyration:")
print(statistics.stdev(radiusOfGyration))
print("Average Distance Between Connected Edges:")
print(sum(averageDistances)/len(averageDistances))
print("Standard Deviation of Distance Between Connected Edges:")
print(statistics.stdev(averageDistances))
print("Number of Triangles:")
print(sum(nx.triangles(mbG).values()) / 3)
print("Average Shortest Path Length:")
print(avgPathLength)
print("Maximum Degree:")
print(maximumDegree)

nx.draw(mbG, nx.get_node_attributes(mbG, 'pos'), node_size=1, width =0.25)
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))

totalJI = 0
trials = 20
otherGraphFile = "" #file name of the graph you want to find the jaccard index of

for i in range(trials):
    with open(otherGraphFile) as f:
        exec(f.read())
    fbEdges = {*mbG.edges(data=False)}
    erIDEdges = {*g.edges(data=False)}
    erEdges = set()
    for edgeTuple in erIDEdges:
        newTuple = (idToZipCode[edgeTuple[0]], idToZipCode[edgeTuple[1]])
        erEdges.add(newTuple)
    ji = len(fbEdges.intersection(erEdges))/len(fbEdges | erEdges)
    totalJI += ji

print("Jaccard Index:")
avgJI = totalJI/trials
print(avgJI)