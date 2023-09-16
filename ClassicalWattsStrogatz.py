import random
import numpy as np
import pandas as pd
import scipy as sp
from scipy import sparse
from scipy.sparse import csr_matrix, rand, triu
import time
import networkx as nx
import math
import matplotlib.pyplot as plt
import statistics
k = 10
p = 1/10

def locInit(file):
    global agents, pop, idToZipCode
    agents, idToZipCode = {}, {}
    agentIdx = 0
    with open (file) as file1:
        locList = [line.strip() for line in file1]
    for loc in locList[1:]:
        loc = loc.split(",")
        agents[agentIdx] = (float(loc[2]), float(loc[1]))
        idToZipCode[agentIdx] = int(loc[0])
        agentIdx+=1
    pop = agentIdx

def generate_wsfriend_matrix():
    m = sparse.lil_matrix((total, total), dtype=np.int8)
    for i in range(total):
        start = i
        for j in range(k +1):
            cf = start + j
            if cf >= total:
                cf = start + j - total
            if cf != i:
                make_friend(m, i, cf)
    for i in range(total):
        start = i
        already_connected = []
        for j in range(k +1):
            cf = start + j
            
            if cf >= total:
                cf = start + j - total
            prob = np.random.random()
            if cf != i and prob <0.5:
                nf = np.random.randint(0, total)
                while nf == i or nf in already_connected:
                    nf = np.random.randint(0, total)
                make_friend(m, i, nf)
                already_connected.append(nf)
                m[i, cf] = 0
                m[cf, i] = 0
                
    return m
def make_friend(m, f1, f2):
    m[f1, f2] = 1
    m[f1, f2] = 1

locInit("VAZipcodesWithLinks.csv")
total = len(agents)
start_time = time.time()


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

for i in range(count):
    print("on iteration number " + str(i))
    fm = generate_wsfriend_matrix()
    end_time = time.time()
    print("it took " + str(round(end_time-start_time)) + " seconds to run")
    cx = sparse.coo_matrix(fm)

    g = nx.Graph()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        g.add_edge(i, j)
    en = time.time()
    nx.set_node_attributes(g, agents, name="pos")

    degrees = g.degree()
    sum_of_edges = sum(dict(degrees).values())
    averageDegree.append(sum_of_edges/len(agents))
    stddevDegree.append(statistics.stdev(dict(degrees).values()))
    numEdges.append(g.number_of_edges())

    avgPathLength.append(nx.average_shortest_path_length(g))

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