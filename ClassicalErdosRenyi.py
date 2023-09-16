import numpy as np
import time
import random
from scipy import sparse
from scipy.stats import uniform
from scipy.sparse import csr_matrix, rand, triu
import matplotlib.pylab as plt
import networkx as nx
import statistics

start_time = time.time()


LEN = 0
FRIENDS = 14
st = time.time()

agentIdx = 0
file = "FairfaxCensusTractLatLong.csv"
g = nx.Graph()
agentIdx = 0
agents = {}
idToZipCode = {}
with open (file) as file1:
    locList = [line.strip() for line in file1]
for loc in locList[1:]:
    loc = loc.split(",")
    g.add_node(agentIdx, pos=(float(loc[2]), float(loc[1])))
    agents[agentIdx] = (float(loc[2]), float(loc[1]))
    idToZipCode[agentIdx] = int(loc[0])
    agentIdx+=1
LEN = agentIdx


def generateSparseMatrix(LEN, x):
    flist = sparse.lil_matrix((LEN, LEN), dtype='int8')
    for i in range(LEN):
        rand_ints = random.sample(range(LEN), x[i])
        for j in rand_ints:
            flist[i, j] = 1
    return flist


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
    print("Iteration " + str(i))
    x = np.random.binomial(n=FRIENDS/0.5, p=0.5, size=LEN)

    m = generateSparseMatrix(LEN, x)
    g = nx.Graph()
    cx = sparse.coo_matrix(m)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        if i != j:
            g.add_edge(i, j)

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
print("--- %s seconds ---" % (time.time() - start_time))
