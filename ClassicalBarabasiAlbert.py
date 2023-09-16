import random
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, rand, triu
import time
import matplotlib.pylab as plt
import networkx as nx
import statistics

start_time = time.time()

#starting number of nodes
m_0 = 10
# starting number of edges per node
num_friends = 10

def locInit(file):
    global agents, pop, idToZipCode
    agents = {}
    idToZipCode = {}
    agentIdx = 0
    with open (file) as file1:
        locList = [line.strip() for line in file1]
    for loc in locList[1:]:
        loc = loc.split(",")
        agents[agentIdx] = (float(loc[2]), float(loc[1]))
        idToZipCode[agentIdx] = int(loc[0])
        agentIdx+=1
    pop = agentIdx

locInit("VAZipcodesWithLinks.csv")


def start_ba_model(matrix, d_list, m_0):
    # list of initial nodes
    f1=[]
    for p in range(m_0):
        f1.append(p)
    # loop through nodes 0 to m_0 and add connections
    for i in range(m_0):
        friend_counter = 0
        while friend_counter < num_friends and friend_counter <(m_0-1):
            x = random.choice(f1)
            if x != i:
                matrix[i,x] = 1
                matrix[x,i] = 1
                d_list.append(i)
                d_list.append(x)
                friend_counter=friend_counter + 1
                f1.remove(x)
        f1.clear()
        for p in range(m_0):
            f1.append(p)
    return (matrix, d_list)
            
    

def run_model (matrix, d_list, m_0, n1):
    matrix, d_list = start_ba_model(matrix, d_list, m_0)
    matrix = add_edges(matrix, d_list, m_0, n1)
    return matrix


def add_edges (matrix, d_list, m_0, n1):
    t = n - m_0
    for i in range(t): 
        friend_counter = 0
        while friend_counter < num_friends:
            n2 = random.choice(d_list)
            if matrix[n2,n1] == 0 and n2 != n1:
                matrix[n1,n2] = 1
                matrix[n2,n1] = 1
                d_list.append(n1)
                d_list.append(n2)
                friend_counter = friend_counter + 1
        n1 = n1+1
    return matrix

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

    n = pop

    n1= m_0

    d_list = []

    matrix = sparse.lil_matrix((n, n), dtype=np.int8)
    matrix = run_model(matrix, d_list, m_0, n1)


    cx = sparse.coo_matrix(matrix)
    g = nx.Graph()

    for i, j, v in zip(cx.row, cx.col, cx.data):
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