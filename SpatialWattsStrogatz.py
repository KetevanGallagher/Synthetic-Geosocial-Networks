
# coding: utf-8


import matplotlib.pyplot as plt
import pylab as pl
import random as rd
import numpy as np
import networkx as nx
import time
import statistics



def init(n,density):
    global agents, pop
    width = int(round((n/density)**0.5))
    height = int(round((n/density)**0.5))
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

def spatialInit(n,density):
    global agents, pop
    width = int(round((n/density)**0.5))
    height = int(round((n/density)**0.5))
    config = np.zeros([height, width])
    pop = 0 #actual population size
    for x in range(height):
        for y in range(width):
            if rd.random() < density:
                config[x,y] = 1
                pop += 1

    seq = [x for x in range(pop)]
    rd.shuffle(seq)
    agents = {}
    z = 0
    for i in range(height):
        for j in range(width):
            if config[i,j] == 1:
                agents[seq[z]]=(i, j)
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


def SpatialSW2(p,k,a):
    global u,v,j,w,G,targets,nodes
    G = nx.Graph()
    nodes = list(range(pop)) # nodes are labeled 0 to n-1
    # connect each node to k/2 neighbors
    for j in range(1, k // 2+1):
        targets = nodes[j:] + nodes[0:j] # first j nodes are now last in list
        G.add_edges_from(zip(nodes,targets))
    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for j in range(1, k // 2+1): # outer loop is neighbors
        targets = nodes[j:] + nodes[0:j] # first j nodes are now last in list
        # inner loop in node order
        for u,v in zip(nodes,targets): 
            if rd.random() < p:
                chosen = False
                while chosen == False:
                    w = rd.choice(nodes)
                    while w == u or G.has_edge(u, w): # Enforce no self-loops or multiple edges
                        w = rd.choice(nodes)
                    u_coord = agents[u]
                    w_coord = agents[w]
                    d = (float(u_coord[0] - w_coord[0])**2 + (float(u_coord[1] - w_coord[1])**2))**0.5
                    q = (d)**(-a)
                    if rd.random() < q:
                        G.remove_edge(u,v)  
                        G.add_edge(u,w)
                        chosen = True
    nx.set_node_attributes(G, agents, name="pos")
    return G



def SpSW(k,a):
    global u,v,j,w,G,targets,nodes
    G = nx.Graph()
    nodes = list(range(pop)) # nodes are labeled 0 to n-1
    # connect each node to k/2 neighbors
    for j in range(1, k // 2+1):
        targets = nodes[j:] + nodes[0:j] # first j nodes are now last in list
        G.add_edges_from(zip(nodes,targets))
    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for j in range(1, k // 2+1): # outer loop is neighbors
        targets = nodes[j:] + nodes[0:j] # first j nodes are now last in list
        # inner loop in node order
        for u,v in zip(nodes,targets): 
            u_coord = agents[u]
            v_coord = agents[v]
            d = (float(u_coord[0] - v_coord[0])**2 + (float(u_coord[1] - v_coord[1])**2))**0.5
            p = (d)**(-a)
            if rd.random() < p:
                w = rd.choice(nodes)
                while w == u or G.has_edge(u, w): # Enforce no self-loops or multiple edges
                    w = rd.choice(nodes)
                G.remove_edge(u,v)
                G.add_edge(u,w)
    nx.set_node_attributes(G, agents, name="pos")
    return G


start_time = time.time()
density=1
#locInit("FairfaxCensusTractLatLong.csv")
locInit("VAZipCodesWithLinks.csv")
n=len(agents)

a = 3
p = 1/10
k = 20

numEdges1 = []
averageDegree1 = []
stddevDegree1 = []
radiusGyration1 = []
stddevRadiusGyration1 = []
avgLengthEdges1 = []
stdDevEdgeLength1 = []
numTriangles1 = []
avgPathLength1 = []
maximumDegree1 = []

numEdges2 = []
averageDegree2 = []
stddevDegree2 = []
radiusGyration2 = []
stddevRadiusGyration2 = []
avgLengthEdges2 = []
stdDevEdgeLength2 = []
numTriangles2 = []
avgPathLength2 = []
maximumDegree2 = []

count = 1

for i in range(count):
    start_time = time.time()
    print("Iteration " + str(i))
    g1 = SpSW(k, a)
    g2 = SpatialSW2(p, k, a)
    n=len(agents)


    degrees = g1.degree()
    sum_of_edges = sum(dict(degrees).values())
    averageDegree1.append(sum_of_edges/pop)
    stddevDegree1.append(statistics.stdev(dict(degrees).values()))
    numEdges1.append(g1.number_of_edges())
    
    degList1 = [i for i in degrees]
    maxDeg = max(degList1, key= lambda x: x[1])
    maximumDegree1.append(maxDeg[1])
    
    avgPathLength1.append(nx.average_shortest_path_length(g1))

    averageDistances = []
    radiusOfGyration = []

    for node in g1:
        maxDistance = 0
        for n in g1.neighbors(node):
            coord1, coord2 = agents[node], agents[n]
            dist = ((111139*float(coord1[0] - coord2[0]))**2 + ((111139*float(coord1[1] - coord2[1])**2)))**0.5
            averageDistances.append(dist)
            if dist > maxDistance:
                maxDistance = dist
        radiusOfGyration.append(maxDistance)

    radiusGyration1.append(sum(radiusOfGyration)/pop)
    stddevRadiusGyration1.append(statistics.stdev(radiusOfGyration))
    avgLengthEdges1.append(sum(averageDistances)/len(averageDistances))
    stdDevEdgeLength1.append(statistics.stdev(averageDistances))

    numTriangles1.append(sum(nx.triangles(g1).values()) / 3)

    degrees = g2.degree()
    sum_of_edges = sum(dict(degrees).values())
    averageDegree2.append(sum_of_edges/pop)
    stddevDegree2.append(statistics.stdev(dict(degrees).values()))
    numEdges2.append(g2.number_of_edges())

    avgPathLength2.append(nx.average_shortest_path_length(g2))

    degList2 = [i for i in degrees]
    maxDeg = max(degList2, key= lambda x: x[1])
    maximumDegree2.append(maxDeg[1])

    averageDistances = []
    radiusOfGyration = []

    for node in g2:
        maxDistance = 0
        for n in g2.neighbors(node):
            coord1, coord2 = agents[node], agents[n]
            dist = ((111139*float(coord1[0] - coord2[0]))**2 + ((111139*float(coord1[1] - coord2[1])**2)))**0.5
            averageDistances.append(dist)
            if dist > maxDistance:
                maxDistance = dist
        radiusOfGyration.append(maxDistance)

    radiusGyration2.append(sum(radiusOfGyration)/pop)
    stddevRadiusGyration2.append(statistics.stdev(radiusOfGyration))
    avgLengthEdges2.append(sum(averageDistances)/len(averageDistances))
    stdDevEdgeLength2.append(statistics.stdev(averageDistances))

    numTriangles2.append(sum(nx.triangles(g2).values()) / 3)

print("Number of edges, Graph1:")
print(sum(numEdges1)/count)
print("Average Degree, Graph1:")
print(sum(averageDegree1)/count)
print("Standard Deviation of the Degree, Graph1:")
print(sum(stddevDegree1)/count)
print("Radius of Gyration, Graph1:")
print(sum(radiusGyration1)/count)
print("Standard Deviation of Radius Gyration, Graph1:")
print(sum(stddevRadiusGyration1)/count)
print("Average Distance Between Connected Edges, Graph1:")
print(sum(avgLengthEdges1)/count)
print("Standard Deviation of Distance Between Connected Edges, Graph1:")
print(sum(stdDevEdgeLength1)/count)
print("Number of Triangles, Graph1:")
print(sum(numTriangles1)/count)

print("Number of edges, Graph2:")
print(sum(numEdges2)/count)
print("Average Degree, Graph2:")
print(sum(averageDegree2)/count)
print("Standard Deviation of the Degree, Graph2:")
print(sum(stddevDegree2)/count)
print("Radius of Gyration, Graph2:")
print(sum(radiusGyration2)/count)
print("Standard Deviation of Radius Gyration, Graph2:")
print(sum(stddevRadiusGyration2)/count)
print("Average Distance Between Connected Edges, Graph2:")
print(sum(avgLengthEdges2)/count)
print("Standard Deviation of Distance Between Connected Edges, Graph2:")
print(sum(stdDevEdgeLength2)/count)
print("Number of Triangles, Graph2:")
print(sum(numTriangles2)/count)
print("Average Shortest Path Length, Graph2:")
print(sum(avgPathLength2)/count)
print("Maximum Degree, Graph2:")
print(sum(maximumDegree2)/count)

nx.draw(g1, nx.get_node_attributes(g1, 'pos'), node_size=1, width=0.25)
plt.show()

nx.draw(g2, nx.get_node_attributes(g2, 'pos'), node_size=1, width=0.25)
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
