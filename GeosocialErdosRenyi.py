import matplotlib.pyplot as plt
import random as rd
import numpy as np
import networkx as nx
import time
import statistics


def locInit(file):
    global agents, pop, idToZipCode
    agents, idToZipCode = {}, {}
    agentIdx = 1
    with open (file) as file1:
        locList = [line.strip() for line in file1]
    for loc in locList[1:]:
        loc = loc.split(",")
        agents[agentIdx] = (float(loc[2]), float(loc[1]))
        idToZipCode[agentIdx] = int(loc[0])
        agentIdx+=1
    pop = agentIdx

def findScalingFactor(net):
    distList = []
    for node in net:
        coord1 = agents[node]
        for nextNode in net:
            if node != nextNode: 
                coord2 = agents[nextNode]
                if node == nextNode:
                    continue
                dist = (float(coord1[0] - coord2[0])**2 + (float(coord1[1] - coord2[1])**2))**0.5
                distList.append(dist)
    distList.sort()
    minD = distList[6805] #change the index for a different scaling factor
    scalingFactor = 1/minD
    return scalingFactor

def Pr_Con(a,scalingFactor,node1, node2):
    coord1 = agents[node1]
    coord2 = agents[node2]
    if node1 == node2:
        return 0
    dist = (float(coord1[0] - coord2[0])**2 + (float(coord1[1] - coord2[1])**2))**0.5
    if dist < 1/scalingFactor:
        dist = 1/scalingFactor
    pr = (dist*scalingFactor)**(-a)
    return pr


def SpRd(a):
    network = nx.Graph()
    for node in range(1,pop):
        coord = agents[node]
        network.add_node(node, pos=(coord[0], coord[1]))
    scalingFactor = findScalingFactor(network)
    for node1 in range(1,pop):
        for node2 in range(node1 + 1,pop):
            chance = rd.random()
            Pr_Connection = Pr_Con(a,scalingFactor,node1,node2)
            if chance < Pr_Connection and not network.has_edge(node1,node2):
                network.add_edge(node1,node2)
                network.add_edge(node2,node1)
    return network

start_time = time.time()

a = 3

locInit("VAZipcodesWithLinks.csv")
#locInit("FairfaxCensusTractLatLong.csv")

numEdges = []
averageDegree = []
stddevDegree = []
radiusGyration = []
stddevRadiusGyration = []
avgLengthEdges = []
stdDevEdgeLength = []
numTriangles = []
avgPathLength = []
maximumDegree = []

count = 1
numTrials = count

while count >0:
    
    g = SpRd(a)

    print("Iteration " + str(numTrials-count))
    print(g)

    degrees = g.degree()
    sum_of_edges = sum(dict(degrees).values())
    averageDegree.append(sum_of_edges/len(agents))
    stddevDegree.append(statistics.stdev(dict(degrees).values()))
    numEdges.append(g.number_of_edges())

    degList = [i for i in degrees]
    maxDeg = max(degList, key= lambda x: x[1])
    maximumDegree.append(maxDeg[1])

    averageDistances = []
    radiusOfGyration = []

    for node in g:
        maxDistance = 0
        for n in g.neighbors(node):
            coord1, coord2 = agents[node], agents[n]
            dist = ((111139*float(coord1[0] - coord2[0]))**2 + ((111139*float(coord1[1] - coord2[1])**2)))**0.5
            averageDistances.append(dist)
            if dist > maxDistance:
                maxDistance = dist
        radiusOfGyration.append(maxDistance)

    radiusGyration.append(sum(radiusOfGyration)/len(agents))
    stddevRadiusGyration.append(statistics.stdev(radiusOfGyration))
    avgLengthEdges.append(sum(averageDistances)/len(averageDistances))
    stdDevEdgeLength.append(statistics.stdev(averageDistances))

    numTriangles.append(sum(nx.triangles(g).values()) / 3)

    count -= 1

count = numTrials

print("Number of Edges:")
print(sum(numEdges)/count)
print("Average Degree:")
print(sum(averageDegree)/count)
print("Standad Deviation of the Degree:")
print(sum(stddevDegree)/count)
print("Radius of Gyration:")
print(sum(radiusGyration)/count)
print("Standard Deviation of Radius of Gyration:")
print(sum(stddevRadiusGyration)/count)
print("Average Distance Between Connected Nodes:")
print(sum(avgLengthEdges)/count)
print("Standard Deviation of Distance Between Connected Nodes:")
print(sum(stdDevEdgeLength)/count)
print("Number of Triangles")
print(sum(numTriangles)/count)
print("Maximum Degree:")
print(sum(maximumDegree)/count)

nx.draw(g, nx.get_node_attributes(g, 'pos'), node_size=1, width=0.25)
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
