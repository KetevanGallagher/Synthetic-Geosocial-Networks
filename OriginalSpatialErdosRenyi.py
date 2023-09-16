import matplotlib.pyplot as plt
import pylab as pl
import random as rd
import numpy as np
import networkx as nx
from scipy import stats
import scipy as sp
import time
import statistics

# Code from Generating and analyzing spatial social networks. Function "locInit" was added so
# real world data could be used. Function "findC" was added to automate finding the normalizing
# coefficient. 

def init(n,density):
    global agents, pop
    height = n//max([h for h in range(1, int(n**0.5)+1) if n%h ==0])
    width = n//height
    config = np.zeros([height, width])
    pop = 0 #actual population size
    for x in range(height):
        for y in range(width):
            if rd.random() < density:
                config[x,y] = 1
                pop += 1

    # Populating the agents matrix with a random sequence of agents
    seq = [x for x in range(pop)]
    rd.shuffle(seq)
    agents = np.zeros([height, width])
    for r in range(height):
        for t in range(width):
            if agents[r,t] == 0:
                agents[r,t] = -1
    z = 0
    for i in range(height):
        for j in range(width):
            if config[i,j] == 1:
                agents[i,j]=seq[z]
                z += 1

def locInit(file):
    global agents, pop, idToZipCode
    agents, idToZipCode = {}, {}
    agents = {}
    agentIdx = 0
    with open (file) as file1:
        locList = [line.strip() for line in file1]
    locList = locList[1:]
    pop = len(locList)
    seq = [x for x in range(pop)]
    rd.shuffle(seq)
    agents = {}
    z = 0
    for agentIdx in seq:
        loc = locList[z]
        loc = loc.split(",")
        agents[agentIdx] = (float(loc[2]), float(loc[1]))
        idToZipCode[agentIdx] = int(loc[0])
        z+=1
    pop = len(agents)

def findC(net, a):
    minD = float("inf")
    maxD = 0
    for node in net:
        coord1 = agents[node]
        for nextNode in net:
            if node != nextNode:
                coord2 = agents[nextNode]
                dist = (float(coord1[0] - coord2[0])**2 + (float(coord1[1] - coord2[1])**2))**0.5
                if dist > maxD:
                    maxD = dist
                elif dist < minD:
                    minD = dist
    c = (1-a)/(maxD**(1-a) - minD**(1-a))
    return c

def Pr_Con(a,c,node1, node2):
    coord1 = agents[node1]
    coord2 = agents[node2]
    dist = (float(coord1[0] - coord2[0])**2 + (float(coord1[1] - coord2[1])**2))**0.5
    pr = c * (dist**(-a))
    return pr


def SpRd(a):
    network = nx.Graph()
    for node in range(1,pop):
        coord = agents[node]
        network.add_node(node, pos=(coord[0], coord[1]))
    c = findC(network, a)
    for node1 in range(1,pop):
        for node2 in range(node1 + 1,pop):
            chance = rd.random()
            Pr_Connection = Pr_Con(a,c,node1,node2)
            if chance < Pr_Connection:
                network.add_edge(node1,node2)
    return network

start_time = time.time()
density=1
a = 3
#locInit("FairfaxCensusTractLatLong.csv")
locInit("VAZipcodesWithLinks.csv")

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
totalTime = 0

count = 1

for i in range(count):
    start_time = time.time()
    print("Iteration " + str(i))
    g = SpRd(a)
    n=len(agents)


    degrees = g.degree()
    sum_of_edges = sum(dict(degrees).values())
    averageDegree.append(sum_of_edges/pop)
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

    radiusGyration.append(sum(radiusOfGyration)/pop)
    stddevRadiusGyration.append(statistics.stdev(radiusOfGyration))
    avgLengthEdges.append(sum(averageDistances)/len(averageDistances))
    stdDevEdgeLength.append(statistics.stdev(averageDistances))

    numTriangles.append(sum(nx.triangles(g).values()) / 3)

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
print("Number of Triangles:")
print(sum(numTriangles)/count)
print("Average Shortest Path Length:")
print(sum(avgPathLength)/count)
print("Maximum Degree:")
print(sum(maximumDegree)/count)


nx.draw(g, nx.get_node_attributes(g, 'pos'), node_size=1, width=0.25)
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))